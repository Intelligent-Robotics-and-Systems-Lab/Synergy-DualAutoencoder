# Usage

The main contribution is from _src.SindyDualAutoencoder.py_, where the Sindy-DualAutoencoder model is created.
There are four main parts to creating this model.
First, it must be imported into your project:

```python
import sys
sys.path.append('/path/to/Synergy-DualAutoencoder/src')
from SindyDualAutoencoder import SindyDualAutoencoderModel
```

Next, create an instance of the Sindy-DualAutoencoder class with the input size of both autoencoders specified:

```python
syda = SindyDualAutoencoderModel(model_input_size1, model_input_size2)
```

Then, specify the desired keras layers for both autoencoders, ensuring that the out size of both encoders are the same:

```python
import tensorflow as tf

# Create encoder and decoder for first autoencoder
syda.createModel1(encode_layers,decode_layers)

# Create encoder and decoder for second autoencoder with same encoder output size as above
syda.createModel2(encode_layers,decode_layers)
```

Finally, the model can be compiled and trained on a dataset:

```python
syda.compile()

# input1, input2 = load_desired_dataset()
syda.fit(self,input1,input2)
```

With this step done, the model can then be used to make predictions:

```python
# test1, test2 = load_desired_testset()
predict(self,test1,encoder=1,decoder=1) # Predicts with input of encoder1 and output of decoder1
predict(self,test1,encoder=1,decoder=2) # Predicts with input of encoder1 and output of decoder2
predict(self,test2,encoder=2,decoder=1) # Predicts with input of encoder2 and output of decoder1
predict(self,test2,encoder=2,decoder=2) # Predicts with input of encoder2 and output of decoder2
```
