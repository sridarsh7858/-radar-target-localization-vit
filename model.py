"""
RD-ViT: a CNN + Vision-Transformer for range-bin LOCALIZATION on IPIX
Range-Doppler maps.  (Localization-only redesign per the guide's instructions.)

Design (the 5 guide changes):
  1. Patch / position embedding -- the CNN backbone is the convolutional patch
     embedder (RD map -> 14 range tokens); a LEARNABLE position embedding is then
     added to the tokens.  (The old range-relative attention bias is removed.)
  2. Standard Multi-Head Attention (keras.layers.MultiHeadAttention).
  3. [DET]/CLS token -- the DETECTOR prepends one learnable [DET] token to the 14
     range tokens (15 tokens total); after the encoder the [DET] token is the global
     summary read by the presence head (canonical ViT classification). The LOCALIZER
     uses only the 14 range tokens (one softmax logit per gate).
  4. MLP (2-layer) blocks for the transformer FFN and the localization head.
  5. TWO INDEPENDENT MODELS (no shared weights): `build_localizer()` (softmax over
     14 range tokens) and `build_detector()` ([DET]-token presence sigmoid). Each has
     its OWN copy of the patch-embed + encoder and is trained separately, so the
     detection and localization objectives never compete.

Input (512 doppler, 14 range, 1 pol = HH).  The whole pipeline is single-channel
HH now -- the dataset, clutter bank and synthetic targets all store only HH.
Outputs:
  build_localizer() -> {"loc_soft": (14,)}  argmax = range bin.
  build_detector()  -> {"presence": (1,)}   sigmoid target-present probability.
Inference rule: target present iff presence >= tau (detector); range bin = argmax(loc).

Note: with absolute position embedding + standard attention the model is no longer
exactly range-equivariant (that was the point of the old range-relative bias).
Range-roll augmentation in train.py makes it see targets at every gate, so it still
learns gate-agnostic localization; test-time roll-averaging becomes an ensemble.
"""
import tensorflow as tf
from tensorflow import keras
from keras import layers

N_GATES = 14
TOKEN_DIM = 128                 # transformer working dim
N_HEADS = 4
N_BLOCKS = 4
FFN_DIM = 256
DROPOUT = 0.15


class AddPositionEmbedding(layers.Layer):
    """Add a learnable position embedding to a (B, tokens, dim) sequence."""

    def build(self, input_shape):
        self.pos = self.add_weight(
            name="pos_emb", shape=(1, input_shape[1], input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.02),
            trainable=True)
        super().build(input_shape)

    def call(self, x):
        return x + self.pos


def vit_block(x, name):
    """Pre-norm ViT block: standard multi-head attention + MLP, both residual."""
    h = layers.LayerNormalization(name=f"{name}_ln1")(x)
    h = layers.MultiHeadAttention(num_heads=N_HEADS, key_dim=TOKEN_DIM // N_HEADS,
                                  dropout=DROPOUT, name=f"{name}_mha")(h, h)
    h = layers.Dropout(DROPOUT, name=f"{name}_drop1")(h)
    x = layers.Add(name=f"{name}_add1")([x, h])
    # MLP (2-layer) -- the channel-mixing feed-forward
    h = layers.LayerNormalization(name=f"{name}_ln2")(x)
    h = layers.Dense(FFN_DIM, activation="gelu", name=f"{name}_mlp1")(h)
    h = layers.Dropout(DROPOUT, name=f"{name}_drop2")(h)
    h = layers.Dense(TOKEN_DIM, name=f"{name}_mlp2")(h)
    return layers.Add(name=f"{name}_add2")([x, h])


class PrependDetToken(layers.Layer):
    """Prepend one learnable [DET] token to a (B, tokens, dim) sequence -> (B, tokens+1, dim).
    The [DET] token attends to all range gates through the encoder and becomes the global
    present/absent summary (canonical ViT [CLS]-token pattern)."""

    def build(self, input_shape):
        self.token = self.add_weight(
            name="det_token", shape=(1, 1, input_shape[2]),
            initializer=keras.initializers.RandomNormal(stddev=0.02), trainable=True)
        super().build(input_shape)

    def call(self, x):
        b = tf.shape(x)[0]
        tok = tf.broadcast_to(self.token, (b, 1, tf.shape(x)[2]))
        return tf.concat([tok, x], axis=1)


def _patch_embed(inp):
    """CNN patch-embedder: RD map -> 14 range tokens of dim 128 (Doppler 512->8)."""
    x = inp
    pools = [(4, 1), (4, 1), (2, 1), (2, 1)]      # 512 / (4*4*2*2) = 8
    chans = [16, 32, 64, 128]
    for i, (pk, ch) in enumerate(zip(pools, chans)):
        x = layers.Conv2D(ch, 3, padding="same", name=f"conv{i}")(x)
        x = layers.BatchNormalization(name=f"bn{i}")(x)
        x = layers.Activation("relu", name=f"relu{i}")(x)
        x = layers.MaxPooling2D(pk, name=f"pool{i}")(x)          # Doppler only
    x = layers.Permute((2, 1, 3), name="to_range_major")(x)      # (14, 8, 128)
    tokens = layers.Reshape((N_GATES, 8 * 128), name="range_tokens")(x)   # (14,1024)
    return layers.Dense(TOKEN_DIM, name="patch_proj")(tokens)    # (14,128)


def _encode(g):
    """Learnable position embedding + N pre-norm ViT blocks -> encoded tokens.
    Auto-sizes to the token count (14 for the localizer, 15 with the [DET] token)."""
    g = AddPositionEmbedding(name="pos_embed")(g)
    g = layers.Dropout(DROPOUT, name="emb_drop")(g)
    for b in range(N_BLOCKS):
        g = vit_block(g, name=f"enc{b}")
    return layers.LayerNormalization(name="enc_out_ln")(g)


def build_localizer(n_pol=1):
    """Independent LOCALIZATION model: patch-embed -> encoder over the 14 range tokens
    -> softmax over the 14 gates. The gates COMPETE via softmax, so the model puts mass
    on ONE gate. No [DET] token -- localization is per-gate, not a global summary.
    n_pol = input polarizations (1 = HH; 4 = HH/HV/VH/VV, the cross-pol lever)."""
    inp = keras.Input(shape=(512, N_GATES, n_pol), name="rd_map")
    g = _encode(_patch_embed(inp))                              # (14,128)
    h = layers.Dropout(DROPOUT, name="loc_drop")(g)
    h = layers.Dense(64, activation="gelu", name="loc_mlp1")(h)
    loc_logits = layers.Dense(1, name="loc_mlp2")(h)             # (14,1)
    loc_logits = layers.Reshape((N_GATES,), name="loc_flat")(loc_logits)
    loc_soft = layers.Softmax(name="loc_soft")(loc_logits)       # (14,) sums to 1
    return keras.Model(inp, {"loc_soft": loc_soft}, name="RD_Localizer")


def build_detector(n_pol=1):
    """Independent DETECTION model: patch-embed -> prepend a learnable [DET] token ->
    encoder over 15 tokens -> the [DET] token feeds the presence sigmoid. The token
    learns WHAT to look for across all gates, replacing fixed avg/max pooling.
    n_pol = input polarizations (1 = HH; 4 = HH/HV/VH/VV, the cross-pol lever)."""
    inp = keras.Input(shape=(512, N_GATES, n_pol), name="rd_map")
    tokens = _patch_embed(inp)                                   # (14,128)
    tokens = PrependDetToken(name="det_prepend")(tokens)        # (15,128)
    g = _encode(tokens)                                          # (15,128)
    cls = layers.Lambda(lambda t: t[:, 0, :], name="take_det")(g)   # (128,) the [DET] token
    p = layers.Dense(64, activation="gelu", name="pres_mlp1")(cls)
    p = layers.Dropout(DROPOUT, name="pres_drop")(p)
    presence = layers.Dense(1, activation="sigmoid", name="presence")(p)  # (1,)
    return keras.Model(inp, {"presence": presence}, name="RD_Detector")


if __name__ == "__main__":
    for build in (build_localizer, build_detector):
        m = build()
        print(f"{m.name}: {m.count_params():,} params")
