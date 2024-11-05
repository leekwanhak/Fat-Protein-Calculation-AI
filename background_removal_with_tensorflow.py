import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

# U-Net 모델을 정의하는 함수
def unet_model(input_size=(256, 256, 3)):
    inputs = Input(input_size)

    # VGG-16을 다운샘플링 파트로 사용 (전이 학습)
    vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=inputs)
    
    # VGG-16의 레이어를 다운샘플링으로 사용
    block1_conv2 = vgg16.get_layer('block1_conv2').output
    block2_conv2 = vgg16.get_layer('block2_conv2').output
    block3_conv3 = vgg16.get_layer('block3_conv3').output
    block4_conv3 = vgg16.get_layer('block4_conv3').output
    block5_conv3 = vgg16.get_layer('block5_conv3').output

    # 업샘플링 (기존 U-Net 구조 적용)
    up6 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(block5_conv3)
    up6 = concatenate([up6, block4_conv3])
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv6)

    up7 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6)
    up7 = concatenate([up7, block3_conv3])
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv7)

    up8 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7)
    up8 = concatenate([up8, block2_conv2])
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv8)

    up9 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8)
    up9 = concatenate([up9, block1_conv2])
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.MeanIoU(num_classes=2)])
    
    return model

# 데이터 증강 설정
data_gen_args = dict(rotation_range=20,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    horizontal_flip=True,
                    fill_mode='nearest')

datagen = ImageDataGenerator(**data_gen_args)

# 예시 원본 데이터셋 로드 (여기서는 임의의 데이터로 가정)
original_images = np.random.rand(120, 256, 256, 3)  # 120개의 원본 이미지
masks = np.random.rand(120, 256, 256, 1)  # 120개의 원본 마스크

# 데이터 증강을 통해 480개의 추가 데이터 생성
augmented_images = []
augmented_masks = []
for img, mask in zip(original_images, masks):
    img = img.reshape((1,) + img.shape)
    mask = mask.reshape((1,) + mask.shape)
    i = 0
    for batch in datagen.flow(img, batch_size=1):
        augmented_images.append(batch[0])
        i += 1
        if i >= 4:  # 각 이미지당 4개의 증강 데이터를 생성하여 총 480개 생성
            break
    j = 0
    for batch in datagen.flow(mask, batch_size=1):
        augmented_masks.append(batch[0])
        j += 1
        if j >= 4:
            break

augmented_images = np.array(augmented_images)
augmented_masks = np.array(augmented_masks)

# 전체 데이터셋 합치기
all_images = np.concatenate((original_images, augmented_images), axis=0)
all_masks = np.concatenate((masks, augmented_masks), axis=0)

# 데이터셋 분할 (6:2:2 비율)
train_size = int(0.6 * len(all_images))
val_size = int(0.2 * len(all_images))
test_size = len(all_images) - train_size - val_size

train_images = all_images[:train_size]
train_masks = all_masks[:train_size]
val_images = all_images[train_size:train_size + val_size]
val_masks = all_masks[train_size:train_size + val_size]
test_images = all_images[train_size + val_size:]
test_masks = all_masks[train_size + val_size:]

# U-Net 모델 생성
model = unet_model()

# 원본 이미지로 모델 학습 및 검증
model.fit(original_images, masks, validation_data=(val_images, val_masks), epochs=10, batch_size=4)

# 원본 이미지 + 증강 데이터로 모델 학습 및 검증
model.fit(train_images, train_masks, validation_data=(val_images, val_masks), epochs=10, batch_size=4)

# 테스트 데이터셋으로 최종 평가
test_loss, test_accuracy, test_iou = model.evaluate(test_images, test_masks)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test IoU: {test_iou:.4f}')
