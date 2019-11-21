# ASSIGNMENT 1 | EIP 4.0 |
---
##### Name: ANKIT SINGH
##### Email: ankitsingh.cool95@gmail.com
##### Github : https://github.com/infinityrun
---

`Logs of 16 epochs`

Train on 60000 samples, validate on 10000 samples
Epoch 1/16

Epoch 00001: LearningRateScheduler setting learning rate to 0.0019.
60000/60000 [==============================] - 13s 213us/step - loss: 0.0327 - acc: 0.9896 - val_loss: 0.0313 - val_acc: 0.9903
Epoch 2/16

Epoch 00002: LearningRateScheduler setting learning rate to 0.001443769.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0250 - acc: 0.9918 - val_loss: 0.0302 - val_acc: 0.9901
Epoch 3/16

Epoch 00003: LearningRateScheduler setting learning rate to 0.0011642157.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0225 - acc: 0.9927 - val_loss: 0.0284 - val_acc: 0.9902
Epoch 4/16

Epoch 00004: LearningRateScheduler setting learning rate to 0.0009753593.
60000/60000 [==============================] - 10s 173us/step - loss: 0.0205 - acc: 0.9933 - val_loss: 0.0210 - val_acc: 0.9935
Epoch 5/16

Epoch 00005: LearningRateScheduler setting learning rate to 0.0008392226.
60000/60000 [==============================] - 10s 171us/step - loss: 0.0180 - acc: 0.9941 - val_loss: 0.0252 - val_acc: 0.9916
Epoch 6/16

Epoch 00006: LearningRateScheduler setting learning rate to 0.0007364341.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0165 - acc: 0.9948 - val_loss: 0.0201 - val_acc: 0.9939
Epoch 7/16

Epoch 00007: LearningRateScheduler setting learning rate to 0.0006560773.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0154 - acc: 0.9948 - val_loss: 0.0202 - val_acc: 0.9936
Epoch 8/16

Epoch 00008: LearningRateScheduler setting learning rate to 0.0005915318.
60000/60000 [==============================] - 10s 172us/step - loss: 0.0140 - acc: 0.9953 - val_loss: 0.0239 - val_acc: 0.9934
Epoch 9/16

Epoch 00009: LearningRateScheduler setting learning rate to 0.0005385488.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0132 - acc: 0.9956 - val_loss: 0.0223 - val_acc: 0.9936
Epoch 10/16

Epoch 00010: LearningRateScheduler setting learning rate to 0.0004942768.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0126 - acc: 0.9957 - val_loss: 0.0219 - val_acc: 0.9930
Epoch 11/16

Epoch 00011: LearningRateScheduler setting learning rate to 0.0004567308.
60000/60000 [==============================] - 10s 170us/step - loss: 0.0123 - acc: 0.9960 - val_loss: 0.0220 - val_acc: 0.9935
Epoch 12/16

Epoch 00012: LearningRateScheduler setting learning rate to 0.0004244861.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0115 - acc: 0.9962 - val_loss: 0.0205 - val_acc: 0.9935
Epoch 13/16

Epoch 00013: LearningRateScheduler setting learning rate to 0.0003964942.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0106 - acc: 0.9965 - val_loss: 0.0197 - val_acc: 0.9943
Epoch 14/16

Epoch 00014: LearningRateScheduler setting learning rate to 0.0003719655.
60000/60000 [==============================] - 10s 168us/step - loss: 0.0097 - acc: 0.9970 - val_loss: 0.0214 - val_acc: 0.9930
Epoch 15/16

Epoch 00015: LearningRateScheduler setting learning rate to 0.000350295.
60000/60000 [==============================] - 10s 169us/step - loss: 0.0091 - acc: 0.9970 - val_loss: 0.0215 - val_acc: 0.9935
Epoch 16/16

Epoch 00016: LearningRateScheduler setting learning rate to 0.0003310105.
60000/60000 [==============================] - 10s 165us/step - loss: 0.0093 - acc: 0.9969 - val_loss: 0.0196 - val_acc: 0.9942

---

`model.evaluate`

Program Score: [0.01956415104058797, 0.9942]

---
`Strategy`

Strategy was very simple to use MMZ in the model and use dropout after maxpool  to get max effect possible.

1> First run 2 kernel and with batch_normalization
2> Max pool the result and again batch_normalization
3> then do dropout of(0.05)
4> repeat step 1,2
5> then do dropout of(0.20)
6> run 2 kernel again to get the final result.

