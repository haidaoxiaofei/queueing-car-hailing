set -x
export DATAPATH=/home/fengchaodavid/ridedispatcher/DeepST
export THEANO_FLAGS="device=gpu,floatX=float32" 
python exptTaxiBJ.py 2
#python exptTaxiBJ.py 4
