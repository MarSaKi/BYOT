

#be your own teach ++
python BYOT.py --gpu 6 --batch_size 64
nohup python -u BYOT.py --gpu 6 --batch_size 128 > train_BYOT.log 2>&1 &


#deep supervised
python DS.py --gpu 6 --batch_size 64
nohup python -u DS.py --gpu 7 --batch_size 128 > train_DS.log 2>&1 &