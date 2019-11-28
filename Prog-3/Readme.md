# Final Validation of base network
> Accuracy on test data is: 82.56

# Model Definition

model = Sequential()
model.add(SeparableConv2D(32, 3, 3, activation='relu', input_shape=(32, 32, 3))) # (30, 3)
model.add(BatchNormalization())
model.add(Dropout(0.12))

model.add(SeparableConv2D(64, kernel_size = (3, 3), strides=(2, 2), activation='relu')) # (14, 5)
model.add(BatchNormalization())
model.add(Dropout(0.13))


model.add(SeparableConv2D(96, 3, 3, activation='relu')) # (12, 7)
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(SeparableConv2D(96, 3, 3, activation='relu')) # (10, 11)
model.add(BatchNormalization())
model.add(Dropout(0.18))

model.add(SeparableConv2D(192, kernel_size = (3, 3), strides=(2, 2), activation='relu')) # (4, 19)
model.add(BatchNormalization())
model.add(Dropout(0.20))

#model.add(SeparableConv2D(192, 3, 3, activation='relu')) # (8, 15)
#model.add(BatchNormalization())
#model.add(Dropout(0.20))


model.add(SeparableConv2D(192, 3, 3, activation='relu')) # (2, 23)
model.add(BatchNormalization())
model.add(Dropout(0.15))


model.add(SeparableConv2D(num_classes, 2, 2, activation='relu')) # (1, 23)

model.add(Flatten())
model.add(Activation('softmax')) #(1, 23)

# 50 epochs

Epoch 1/50
390/390 [==============================] - 43s 111ms/step - loss: 1.6186 - acc: 0.4225 - val_loss: 1.3735 - val_acc: 0.5201
Epoch 2/50
390/390 [==============================] - 15s 38ms/step - loss: 1.2581 - acc: 0.5574 - val_loss: 1.1275 - val_acc: 0.6068
Epoch 3/50
390/390 [==============================] - 15s 38ms/step - loss: 1.1356 - acc: 0.6011 - val_loss: 1.0811 - val_acc: 0.6229
Epoch 4/50
390/390 [==============================] - 15s 38ms/step - loss: 1.0472 - acc: 0.6320 - val_loss: 0.9864 - val_acc: 0.6523
Epoch 5/50
390/390 [==============================] - 15s 38ms/step - loss: 0.9799 - acc: 0.6572 - val_loss: 1.0075 - val_acc: 0.6528
Epoch 6/50
390/390 [==============================] - 15s 37ms/step - loss: 0.9317 - acc: 0.6727 - val_loss: 0.8840 - val_acc: 0.6926
Epoch 7/50
390/390 [==============================] - 15s 37ms/step - loss: 0.8871 - acc: 0.6910 - val_loss: 0.8874 - val_acc: 0.6874
Epoch 8/50
390/390 [==============================] - 15s 38ms/step - loss: 0.8455 - acc: 0.7040 - val_loss: 0.8699 - val_acc: 0.6985
Epoch 9/50
390/390 [==============================] - 15s 38ms/step - loss: 0.8214 - acc: 0.7107 - val_loss: 0.8216 - val_acc: 0.7175
Epoch 10/50
390/390 [==============================] - 15s 38ms/step - loss: 0.7973 - acc: 0.7206 - val_loss: 0.8252 - val_acc: 0.7182
Epoch 11/50
390/390 [==============================] - 15s 37ms/step - loss: 0.7787 - acc: 0.7245 - val_loss: 0.7957 - val_acc: 0.7243
Epoch 12/50
390/390 [==============================] - 15s 37ms/step - loss: 0.7547 - acc: 0.7371 - val_loss: 0.7868 - val_acc: 0.7303
Epoch 13/50
390/390 [==============================] - 15s 38ms/step - loss: 0.7417 - acc: 0.7383 - val_loss: 0.7783 - val_acc: 0.7331
Epoch 14/50
390/390 [==============================] - 15s 37ms/step - loss: 0.7227 - acc: 0.7440 - val_loss: 0.7700 - val_acc: 0.7342
Epoch 15/50
390/390 [==============================] - 15s 38ms/step - loss: 0.7081 - acc: 0.7522 - val_loss: 0.7735 - val_acc: 0.7339
Epoch 16/50
390/390 [==============================] - 15s 38ms/step - loss: 0.7000 - acc: 0.7562 - val_loss: 0.7450 - val_acc: 0.7399
Epoch 17/50
390/390 [==============================] - 15s 38ms/step - loss: 0.6811 - acc: 0.7603 - val_loss: 0.7606 - val_acc: 0.7376
Epoch 18/50
390/390 [==============================] - 15s 39ms/step - loss: 0.6718 - acc: 0.7636 - val_loss: 0.8031 - val_acc: 0.7316
Epoch 19/50
390/390 [==============================] - 15s 38ms/step - loss: 0.6595 - acc: 0.7686 - val_loss: 0.7938 - val_acc: 0.7366
Epoch 20/50
390/390 [==============================] - 15s 38ms/step - loss: 0.6490 - acc: 0.7701 - val_loss: 0.7460 - val_acc: 0.7495
Epoch 21/50
390/390 [==============================] - 15s 38ms/step - loss: 0.6337 - acc: 0.7764 - val_loss: 0.7202 - val_acc: 0.7577
Epoch 22/50
390/390 [==============================] - 15s 37ms/step - loss: 0.6293 - acc: 0.7772 - val_loss: 0.7403 - val_acc: 0.7466
Epoch 23/50
390/390 [==============================] - 15s 37ms/step - loss: 0.6237 - acc: 0.7793 - val_loss: 0.7319 - val_acc: 0.7556
Epoch 24/50
390/390 [==============================] - 15s 38ms/step - loss: 0.6114 - acc: 0.7847 - val_loss: 0.7142 - val_acc: 0.7570
Epoch 25/50
390/390 [==============================] - 15s 37ms/step - loss: 0.6084 - acc: 0.7836 - val_loss: 0.7208 - val_acc: 0.7545
Epoch 26/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5978 - acc: 0.7898 - val_loss: 0.7206 - val_acc: 0.7545
Epoch 27/50
390/390 [==============================] - 15s 37ms/step - loss: 0.5886 - acc: 0.7912 - val_loss: 0.7605 - val_acc: 0.7470
Epoch 28/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5872 - acc: 0.7913 - val_loss: 0.7142 - val_acc: 0.7604
Epoch 29/50
390/390 [==============================] - 14s 37ms/step - loss: 0.5734 - acc: 0.7964 - val_loss: 0.7267 - val_acc: 0.7551
Epoch 30/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5752 - acc: 0.7968 - val_loss: 0.6913 - val_acc: 0.7648
Epoch 31/50
390/390 [==============================] - 15s 37ms/step - loss: 0.5630 - acc: 0.8005 - val_loss: 0.7277 - val_acc: 0.7571
Epoch 32/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5613 - acc: 0.8016 - val_loss: 0.7105 - val_acc: 0.7639
Epoch 33/50
390/390 [==============================] - 15s 37ms/step - loss: 0.5599 - acc: 0.8018 - val_loss: 0.6965 - val_acc: 0.7623
Epoch 34/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5513 - acc: 0.8051 - val_loss: 0.7339 - val_acc: 0.7553
Epoch 35/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5538 - acc: 0.8052 - val_loss: 0.7074 - val_acc: 0.7623
Epoch 36/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5465 - acc: 0.8080 - val_loss: 0.6950 - val_acc: 0.7652
Epoch 37/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5367 - acc: 0.8089 - val_loss: 0.6785 - val_acc: 0.7670
Epoch 38/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5295 - acc: 0.8127 - val_loss: 0.7072 - val_acc: 0.7632
Epoch 39/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5256 - acc: 0.8145 - val_loss: 0.6890 - val_acc: 0.7703
Epoch 40/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5240 - acc: 0.8133 - val_loss: 0.6846 - val_acc: 0.7740
Epoch 41/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5194 - acc: 0.8159 - val_loss: 0.6903 - val_acc: 0.7696
Epoch 42/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5202 - acc: 0.8162 - val_loss: 0.6873 - val_acc: 0.7736
Epoch 43/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5121 - acc: 0.8193 - val_loss: 0.7016 - val_acc: 0.7692
Epoch 44/50
390/390 [==============================] - 15s 37ms/step - loss: 0.5086 - acc: 0.8183 - val_loss: 0.6776 - val_acc: 0.7768
Epoch 45/50
390/390 [==============================] - 15s 38ms/step - loss: 0.5045 - acc: 0.8206 - val_loss: 0.6955 - val_acc: 0.7690
Epoch 46/50
390/390 [==============================] - 15s 37ms/step - loss: 0.5043 - acc: 0.8214 - val_loss: 0.6716 - val_acc: 0.7775
Epoch 47/50
390/390 [==============================] - 15s 38ms/step - loss: 0.4922 - acc: 0.8232 - val_loss: 0.7199 - val_acc: 0.7643
Epoch 48/50
390/390 [==============================] - 15s 38ms/step - loss: 0.4948 - acc: 0.8229 - val_loss: 0.6833 - val_acc: 0.7775
Epoch 49/50
390/390 [==============================] - 15s 38ms/step - loss: 0.4990 - acc: 0.8229 - val_loss: 0.6951 - val_acc: 0.7701
Epoch 50/50
390/390 [==============================] - 15s 37ms/step - loss: 0.4920 - acc: 0.8254 - val_loss: 0.6906 - val_acc: 0.7744
