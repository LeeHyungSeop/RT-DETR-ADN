"""by lyuwenyu
"""

import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS


def main(args, ) -> None:
    '''main
    '''
    dist.init_distributed()

    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    # 2024.05.15 @hslee
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    '''
        TASKS is Dictionary datatype
            [key:'detection', value:DetSolver]
        
        cfg.yaml_cfg : the configuration file path (--config)
            (default) configs/rtdetr/rtdetr_r50vd_6x_coco.yml
        'task' : detection 
            (by 'configs/rtdetr/rtdetr_r50vd_6x_coco.yml' > __include__ > './include/rtdetr_r50vd.yml' > 'task' : detection)
        -> TASKS['detection'](cfg)
        -> DetSolver(cfg)
        
        det_solver.py > class DetSolover(cfg)
    '''
    
    
    if args.test_only:
        solver.val() # validation
    else:
        solver.fit() # training


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, )
    parser.add_argument('--resume', '-r', type=str, )
    parser.add_argument('--tuning', '-t', type=str, )
    parser.add_argument('--test-only', action='store_true', default=False,)
    parser.add_argument('--amp', action='store_true', default=False,)

    args = parser.parse_args()

    main(args)


'''
pip install -r requirements.txt

# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    2>&1 | tee ./logs/resnetADN_super.txt
    
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml \
    --resume "/home/hslee/Desktop/INU_RISE/06_RT-DETR-ADN/output/rtdetr_r50vd_6x_coco_resnetADN_super/checkpoint.pth" \
    2>&1 | tee -a ./logs/resnetADN_super.txt    

# train


# test on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml --test-only --resume "/home/hslee/Desktop/Embedded_AI/INU_4-1/RISE/06_RT-DETR-ADN/output/rtdetr_r50vd_6x_coco/checkpoint0000.pth" \
    2>&1 | tee ./logs/test.txt

'''