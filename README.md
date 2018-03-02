# OpenAI-XOR
Warmup #1 for OpenAI's Request for Research 2.0 https://blog.openai.com/requests-for-research-2

## Objective
Train an LSTM neural network to solve the XOR problem i.e. determine the parity of a given sequence of bits. The LSTM model should consume the bits one by one and output the parity. I am going by even parity bit: The number of 1's are counted and if the count is odd, parity is 1. If the count is even then the parity is 0. OpenAI defines 2 tasks to complete:

1. Generate a dataset of random 100,000 binary strings of length 50. Train the LSTM; what performance do you get?
2. Generate a dataset of random 100,000 binary strings, where the length of each string is independently and randomly chosen between 1 and 50. Train the LSTM. Does it succeed? What explains the difference?

# Results
I used Keras to build the LSTM model and numpy to generate random training/testing data. I used a model with 1 LSTM layer (32 memory units, tanh activation) and 1 densely connected output layer (sigmoid activation since this is a binary classification problem). I also use a generic adam optimizer and binary cross entropy loss function. Trained over 20 epochs with a batch size of 32 I get the following results

|               | 'pre' Padding  | 'post' Padding | Accuracy | 
| ------------- |:--------------:| :-------------:| --------:|
| Fixed Length    | **N/A**      | **N/A**        | 50.40%   |
| Variable Length | Yes          | No             | 100%     |
| Variable Length | No           | Yes            | 49.86%   |

'pre' padding means we pad the sequences with 0's before our main sequence starts while 'post' padding means we add padding after the sequence. **0000**1101 vs. 1101**0000**. Variable length with 'pre' padding solves the problem with 100% validation and test accuracy. However with 'post' padding the accuracy is lower than for fixed length. I think this may be because with the padding inserted after our sequence, then the lstm has to 'remember' the beginning of the sequence (which the most important part of the sequence in this case). When the padding is inserted before the sequence then the meat of the sequence, i.e. the 1's, are the last part of the sequence to be fed into the lstm model during training and it is 'fresh' in the lstm's memory. With the fixed sequence length of 50 the 1's are spread out throught the sequence and in some cases the lstm has to remember the entire sequence to know how many 1's appear.

## Usage
This program requires numpy, keras, and matplotlib. To use it navigate to the directory where `xor_nn.py` file is located and run the following code in terminal/command line.

`python xor_nn.py -l 50` --> For fixed length of 50

`python xor_nn.py -l -1` --> For random variable length for each binary sequence
