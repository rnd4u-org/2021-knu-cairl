def build_model():
    from tensorflow.keras.applications.inception_v3 import InceptionV3
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D

    base_model = InceptionV3(weights='imagenet', include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    x = Dense(1024, activation='relu')(x)
    x = Dense(512, activation='relu')(x)
    x = Dense(128, activation='relu')(x)
    predictions = Dense(2, activation='softmax')(x)

    return Model(inputs=base_model.input, outputs=predictions)