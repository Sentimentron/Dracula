export THEANO_FLAGS="floatX=float32,device=gpu,lib.cnmem=0.7,nvcc.fastmath=True"
# Initially, no LSTM layers
python lstm.py

python lstm.py --letters 1 --model lstm_model.npz

python lstm.py --letters 1 --words 1 --model lstm_model.npz
python lstm.py --letters 1 --words 2 --model lstm_model.npz
python lstm.py --letters 1 --words 3 --model lstm_model.npz
python lstm.py --letters 2 --words 3 --model lstm_model.npz
#ython lstm.py --letters 2 --words 4 --model lstm_model.npz
#ython lstm.py --letters 2 --words 5 --model lstm_model.npz
#ython lstm.py --letters 3 --words 5 --model lstm_model.npz
#ython lstm.py --letters 3 --words 6 --model lstm_model.npz

#THEANO_FLAGS="floatX=float32,device=gpu" python lstm.py --stage 2 --model lstm_model.npz
