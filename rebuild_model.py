from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

# Rebuild SAME architecture
input_layer = Input(shape=(187,))

x = Dense(128, activation='relu')(input_layer)
x = Dropout(0.2)(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(32, activation='relu')(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.2)(x)

x = Dense(128, activation='relu')(x)

output_layer = Dense(187, activation='sigmoid')(x)

model = Model(inputs=input_layer, outputs=output_layer)

model.load_weights("ecg_autoencoder.keras")

# Save clean model
model.save("model.h5")

print("Clean model saved as model.h5")