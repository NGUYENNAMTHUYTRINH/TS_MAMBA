import argparse
import torch
from experiments.exp_long_term_forecasting import Exp_Long_Term_Forecast
import random
import numpy as np


def str2bool(value):
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes", "y"}


if __name__ == '__main__':
    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Transformer - Air Quality Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='1=train, 0=test only')
    parser.add_argument('--model_id', type=str, required=True, default='air_quality', help='model id')
    parser.add_argument('--model', type=str, default='Transformer', choices=['Transformer'],
                        help='model name (only Transformer is supported)')

    # data loader
    parser.add_argument('--data', type=str, default='air_quality', help='dataset type (air_quality)')
    parser.add_argument('--root_path', type=str, default='./dataset/air_quality/',
                        help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='air_quality.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='M: multivariate->multivariate, S: univariate, MS: multivariate->univariate')
    parser.add_argument('--target', type=str, default='aqi',
                        help='target feature for S or MS task (e.g. aqi, pm25, co2)')
    parser.add_argument('--freq', type=str, default='h',
                        help='time features freq: s/t/h/d/b/w/m')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/',
                        help='location of model checkpoints')

    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction length')

    # model define
    parser.add_argument('--enc_in', type=int, default=11,
                        help='encoder input size. For one-location air_quality: '
                             'use 11 for features=M/MS, or 1 for features=S.')
    parser.add_argument('--dec_in', type=int, default=11, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=11, help='output size')
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=512, help='dimension of FCN')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='do NOT use distilling in encoder', default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding: timeF / fixed / learned')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true',
                        help='output attention weights')
    parser.add_argument('--do_predict', action='store_true',
                        help='predict unseen future data')
    parser.add_argument('--skip_test', action='store_true',
                        help='train only; do not run test/future forecast after training')

    # optimization
    parser.add_argument('--num_workers', type=int, default=0, help='dataloader num workers')
    parser.add_argument('--itr', type=int, default=1, help='number of experiment repeats')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0005, help='learning rate')
    parser.add_argument('--des', type=str, default='Exp', help='experiment description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='lr adjustment strategy')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=str2bool, default=True, help='use GPU if available')
    parser.add_argument('--gpu', type=int, default=0, help='GPU id')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False,
                        help='use multiple GPUs')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='GPU device ids')

    # Transformer compatibility args
    parser.add_argument('--exp_name', type=str, default='MTSF',
                        help='experiment name: MTSF (multivariate time series forecasting)')
    parser.add_argument('--channel_independence', type=bool, default=False,
                        help='channel independence mechanism')
    parser.add_argument('--inverse', action='store_true', default=False,
                        help='inverse output data')
    parser.add_argument('--class_strategy', type=str, default='projection',
                        help='projection / average / cls_token')
    parser.add_argument('--use_norm', type=int, default=1, help='use instance normalization')

    # Unused legacy args (kept for compatibility with exp code)
    parser.add_argument('--target_root_path', type=str, default='./dataset/air_quality/')
    parser.add_argument('--target_data_path', type=str, default='air_quality.csv')
    parser.add_argument('--efficient_training', type=bool, default=False)
    parser.add_argument('--partial_start_index', type=int, default=0)

    args = parser.parse_args()
    if args.data == 'air_quality':
        air_quality_dim = 11
        if args.features == 'S':
            args.enc_in = 1
            args.dec_in = 1
            args.c_out = 1
        elif args.features == 'MS':
            args.enc_in = air_quality_dim
            args.dec_in = 1
            args.c_out = 1
        elif args.features == 'M':
            args.enc_in = air_quality_dim
            args.dec_in = air_quality_dim
            args.c_out = air_quality_dim

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(x) for x in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                args.model_id, args.model, args.data,
                args.features, args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers,
                args.d_ff, args.factor, args.embed, args.distil,
                args.des, ii)

            exp = Exp(args)
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            if not args.skip_test:
                print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.test(setting)

            if args.do_predict and not args.skip_test:
                print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
                exp.predict(setting, True)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
            args.model_id, args.model, args.data,
            args.features, args.seq_len, args.label_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers,
            args.d_ff, args.factor, args.embed, args.distil,
            args.des, ii)

        exp = Exp(args)
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
