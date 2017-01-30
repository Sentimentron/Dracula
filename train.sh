export THEANO_FLAGS="floatX=float32,device=gpu,lib.cnmem=0.7,nvcc.fastmath=True"
# Initially, no LSTM layers
#python lstm.py

python lstm.py --words 1 --model lstm_model.npz
#python lstm.py --words 2 --model lstm_model.npz
#python lstm.py --words 3 --model lstm_model.npz

