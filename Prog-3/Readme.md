# Final Validation of base network
> Accuracy on test data is: 82.56

# Model Definition

model = Sequential()

model.add(SeparableConv2D(48,kernel_size=3, activation='relu', use_bias =False, padding='same',input_shape=(32,32,3))) #48 3
model.add(BatchNormalization())
model.add(Dropout(0.12))

model.add(SeparableConv2D(96,kernel_size=3, activation='relu', use_bias =False,padding='same')) #96 5
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(SeparableConv2D(192,kernel_size=3, activation='relu', use_bias =False,padding='same')) #192 7
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(MaxPooling2D(2,2)) #100 8
model.add(BatchNormalization())
model.add(Dropout(0.12))

model.add(SeparableConv2D(48,kernel_size=3, activation='relu', use_bias =False,padding='same')) #48 12
model.add(BatchNormalization())
model.add(Dropout(0.14))

model.add(SeparableConv2D(96,kernel_size=3, activation='relu', use_bias =False,padding='same')) #96 16
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(SeparableConv2D(192,kernel_size=3, activation='relu', use_bias =False,padding='same')) #192 26
model.add(BatchNormalization())
model.add(Dropout(0.15))

model.add(MaxPooling2D(2,2)) #192 27
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(48,kernel_size=3, activation='relu', use_bias =False,padding='same')) #48 31
model.add(BatchNormalization())
model.add(Dropout(0.1))

model.add(SeparableConv2D(num_classes,kernel_size=1, activation='relu', use_bias =False,padding='same')) #10 38

model.add(GlobalAveragePooling2D()) #10
model.add(Dense(64, activation='relu', use_bias =False)) #64
model.add(Dense(num_classes, activation='softmax', use_bias =False)) #10

# 50 epochs

Epoch 1/50
390/390 [==============================] - 34s 88ms/step - loss: 1.6321 - acc: 0.3715 - val_loss: 1.4591 - val_acc: 0.4773
Epoch 2/50
390/390 [==============================] - 30s 78ms/step - loss: 1.1675 - acc: 0.5740 - val_loss: 1.2157 - val_acc: 0.5702
Epoch 3/50
390/390 [==============================] - 30s 77ms/step - loss: 0.9975 - acc: 0.6386 - val_loss: 1.2945 - val_acc: 0.5615
Epoch 4/50
390/390 [==============================] - 30s 77ms/step - loss: 0.9015 - acc: 0.6743 - val_loss: 1.0813 - val_acc: 0.6190
Epoch 5/50
390/390 [==============================] - 30s 77ms/step - loss: 0.8321 - acc: 0.7005 - val_loss: 0.9930 - val_acc: 0.6505
Epoch 6/50
390/390 [==============================] - 30s 77ms/step - loss: 0.7773 - acc: 0.7227 - val_loss: 0.9053 - val_acc: 0.6861
Epoch 7/50
390/390 [==============================] - 30s 77ms/step - loss: 0.7276 - acc: 0.7414 - val_loss: 0.9267 - val_acc: 0.6869
Epoch 8/50
390/390 [==============================] - 30s 77ms/step - loss: 0.6901 - acc: 0.7548 - val_loss: 0.7584 - val_acc: 0.7307
Epoch 9/50
390/390 [==============================] - 30s 77ms/step - loss: 0.6597 - acc: 0.7652 - val_loss: 0.7971 - val_acc: 0.7230
Epoch 10/50
390/390 [==============================] - 30s 77ms/step - loss: 0.6323 - acc: 0.7766 - val_loss: 0.7625 - val_acc: 0.7361
Epoch 11/50
390/390 [==============================] - 30s 77ms/step - loss: 0.6077 - acc: 0.7856 - val_loss: 0.7098 - val_acc: 0.7548
Epoch 12/50
390/390 [==============================] - 30s 77ms/step - loss: 0.5831 - acc: 0.7940 - val_loss: 0.6452 - val_acc: 0.7794
Epoch 13/50
390/390 [==============================] - 30s 77ms/step - loss: 0.5642 - acc: 0.8002 - val_loss: 0.6773 - val_acc: 0.7700
Epoch 14/50
390/390 [==============================] - 30s 77ms/step - loss: 0.5442 - acc: 0.8091 - val_loss: 0.6665 - val_acc: 0.7742
Epoch 15/50
390/390 [==============================] - 30s 77ms/step - loss: 0.5247 - acc: 0.8152 - val_loss: 0.6565 - val_acc: 0.7778
Epoch 16/50
390/390 [==============================] - 30s 78ms/step - loss: 0.5229 - acc: 0.8144 - val_loss: 0.6571 - val_acc: 0.7855
Epoch 17/50
390/390 [==============================] - 30s 77ms/step - loss: 0.5090 - acc: 0.8193 - val_loss: 0.6836 - val_acc: 0.7673
Epoch 18/50
390/390 [==============================] - 30s 78ms/step - loss: 0.4924 - acc: 0.8270 - val_loss: 0.5950 - val_acc: 0.7980
Epoch 19/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4843 - acc: 0.8284 - val_loss: 0.5681 - val_acc: 0.8071
Epoch 20/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4724 - acc: 0.8338 - val_loss: 0.6098 - val_acc: 0.7993
Epoch 21/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4662 - acc: 0.8348 - val_loss: 0.6690 - val_acc: 0.7750
Epoch 22/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4514 - acc: 0.8419 - val_loss: 0.6115 - val_acc: 0.7941
Epoch 23/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4487 - acc: 0.8424 - val_loss: 0.6407 - val_acc: 0.7895
Epoch 24/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4395 - acc: 0.8466 - val_loss: 0.5694 - val_acc: 0.8092
Epoch 25/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4275 - acc: 0.8510 - val_loss: 0.5728 - val_acc: 0.8100
Epoch 26/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4229 - acc: 0.8509 - val_loss: 0.5879 - val_acc: 0.8043
Epoch 27/50
390/390 [==============================] - 30s 78ms/step - loss: 0.4181 - acc: 0.8545 - val_loss: 0.5613 - val_acc: 0.8181
Epoch 28/50
390/390 [==============================] - 30s 77ms/step - loss: 0.4146 - acc: 0.8512 - val_loss: 0.5512 - val_acc: 0.8141
Epoch 29/50
390/390 [==============================] - 30s 78ms/step - loss: 0.4013 - acc: 0.8579 - val_loss: 0.5669 - val_acc: 0.8079
Epoch 30/50
390/390 [==============================] - 30s 78ms/step - loss: 0.4002 - acc: 0.8603 - val_loss: 0.5369 - val_acc: 0.8239
Epoch 31/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3928 - acc: 0.8605 - val_loss: 0.5255 - val_acc: 0.8252
Epoch 32/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3846 - acc: 0.8642 - val_loss: 0.5723 - val_acc: 0.8124
Epoch 33/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3810 - acc: 0.8641 - val_loss: 0.5750 - val_acc: 0.8154
Epoch 34/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3803 - acc: 0.8646 - val_loss: 0.5188 - val_acc: 0.8266
Epoch 35/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3723 - acc: 0.8676 - val_loss: 0.5323 - val_acc: 0.8215
Epoch 36/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3682 - acc: 0.8708 - val_loss: 0.5429 - val_acc: 0.8196
Epoch 37/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3609 - acc: 0.8727 - val_loss: 0.6169 - val_acc: 0.8011
Epoch 38/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3609 - acc: 0.8727 - val_loss: 0.5660 - val_acc: 0.8137
Epoch 39/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3596 - acc: 0.8724 - val_loss: 0.5196 - val_acc: 0.8327
Epoch 40/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3508 - acc: 0.8750 - val_loss: 0.5149 - val_acc: 0.8299
Epoch 41/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3473 - acc: 0.8756 - val_loss: 0.5407 - val_acc: 0.8264
Epoch 42/50
390/390 [==============================] - 30s 77ms/step - loss: 0.3445 - acc: 0.8771 - val_loss: 0.5215 - val_acc: 0.8287
Epoch 43/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3362 - acc: 0.8805 - val_loss: 0.5425 - val_acc: 0.8265
Epoch 44/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3393 - acc: 0.8790 - val_loss: 0.5262 - val_acc: 0.8250
Epoch 45/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3330 - acc: 0.8827 - val_loss: 0.5281 - val_acc: 0.8315
Epoch 46/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3302 - acc: 0.8827 - val_loss: 0.5439 - val_acc: 0.8236
Epoch 47/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3237 - acc: 0.8858 - val_loss: 0.5532 - val_acc: 0.8235
Epoch 48/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3259 - acc: 0.8841 - val_loss: 0.5168 - val_acc: 0.8298
Epoch 49/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3228 - acc: 0.8844 - val_loss: 0.5213 - val_acc: 0.8314
Epoch 50/50
390/390 [==============================] - 30s 78ms/step - loss: 0.3184 - acc: 0.8862 - val_loss: 0.5445 - val_acc: 0.8253
