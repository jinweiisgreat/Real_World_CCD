import argparse
from project_utils.general_utils import init_experiment, get_mean_lr, str2bool, get_dino_head_weights, accuracy
from config import exp_root, dino_pretrain_path
def get_arguments():
    parser = argparse.ArgumentParser(
                description='cluster',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch_size', default=256, type=int) # 128 for cifar100,cifar10; 256 for imagenet
    # parser.add_argument('--batch_size', default=256, type=int)
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--eval_funcs', nargs='+', help='Which eval functions to use', default=['v1', 'v2', 'ucd'])

    parser.add_argument('--warmup_model_dir', type=str, default=None)
    parser.add_argument('--model_name', type=str, default='vit_dino', help='Format is {model_name}_{pretrain}')
    parser.add_argument('--model_arch', type=str, default='vit_base', help='which model to use')
    parser.add_argument('--dataset_name', type=str, default='cifar100', help='options: cifar10, cifar100, scars')
    parser.add_argument('--prop_train_labels', type=float, default=0.5)
    parser.add_argument('--use_ssb_splits', type=str2bool, default=False)

    parser.add_argument('--grad_from_block', type=int, default=11)
    # parser.add_argument('--lr', type=float, default=0.001)#原来
    parser.add_argument('--lr', type=float, default=0.05)#目前效果不错

    parser.add_argument('--save_best_thresh', type=float, default=None)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--exp_root', type=str, default=exp_root)
    parser.add_argument('--transform', type=str, default='imagenet')
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--base_model', type=str, default='vit_dino')
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--sup_con_weight', type=float, default=0.35)
    parser.add_argument('--n_views', default=2, type=int)
    parser.add_argument('--contrast_unlabel_only', type=str2bool, default=False)

    parser.add_argument('--use_faiss', default=False, type=str2bool, help='use faiss for faster kmeans')
    parser.add_argument('--use_sskmeans', default=False, type=str2bool, help='use sskmeans')
    parser.add_argument('--use_mae', default=False, type=str2bool, help='use mae for pretrain')
    parser.add_argument('--use_ucd', default=False, type=str2bool, help='use ucd for everything')

    parser.add_argument('--print_freq', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=5, type=int)


    parser.add_argument('--pi_prior', default='uniform', type=str, help='')
    parser.add_argument('--prior_dir_counts', default=0.1, type=float)
    parser.add_argument('--prior_mu_0', default='data_mean', type=str)
    parser.add_argument('--prior_sigma_choice', default='isotropic', type=str)
    parser.add_argument('--prior_sigma_scale', default=.005, type=float)
    parser.add_argument('--prior_kappa', default=0.0001, type=float)
    parser.add_argument('--prior_nu', default=769, type=int)

    parser.add_argument('--enable_pcl', default=False, type=str2bool, )
    parser.add_argument('--proto_consistency', default=False, type=str2bool)
    parser.add_argument('--pcl_update_interval', default=1, type=int)
    parser.add_argument('--use_all_proto', default=True, type=str2bool)
    parser.add_argument('--pcl_only_unlabel', default=True, type=str2bool)
    parser.add_argument('--supcon_pcl', default=False, type=str2bool)
    parser.add_argument('--pcl_weight', default=1.0, type=float)

    parser.add_argument('--enable_proto_pair', default=False, type=str2bool)
    parser.add_argument('--pair_proto_num_multiplier', default=3, type=int)

    parser.add_argument('--topk', default=10, type=int)

    parser.add_argument('--evaluate_with_proto', default=False, type=str2bool)
    parser.add_argument('--disable_kmeans_eval', default=False, type=str2bool)

    parser.add_argument('--use_pca_testing', default=False, type=str2bool)

    # NOTE: strong and weak aug
    parser.add_argument('--use_strong_aug', default=False, type=str2bool)

    parser.add_argument('--momentum_proto', default=False, type=str2bool)
    parser.add_argument('--momentum_proto_weight', default=0.99, type=float)

    parser.add_argument('--me_max', default=False, type=str2bool)

    parser.add_argument('--exp_name', default=None, type=str)

    parser.add_argument('--plabel_correct', default=False, type=str2bool)
    parser.add_argument('--plabel_conf_thr', default=0.8, type=float)
    parser.add_argument('--plabel_metric_k_num', default=10, type=int)

    parser.add_argument('--debug', action='store_true',)


    parser.add_argument("--root", type=str, default="", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="", help="output directory")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="checkpoint directory (from which the training resumes)",
    )
    # parser.add_argument(
    #     "--seed", type=int, default=-1, help="only positive value enables a fixed seed"
    # )
    parser.add_argument(
        "--source-domains", type=str, nargs="+", help="source domains for DA/DG"
    )
    parser.add_argument(
        "--target-domains", type=str, nargs="+", help="target domains for DA/DG"
    )
    parser.add_argument(
        "--transforms", type=str, nargs="+", help="data augmentation methods"
    )
    parser.add_argument(
        "--config-file", type=str, default="", help="path to config file"
    )
    parser.add_argument(
        "--dataset-config-file",
        type=str,
        default="",
        help="path to config file for dataset setup",
    )
    parser.add_argument("--trainer", type=str, default="", help="name of trainer")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="",
        help="load model from this directory for eval-only mode",
    )
    parser.add_argument(
        "--load-epoch", type=int, help="load model weights at this epoch for evaluation"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="do not call trainer.train()"
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line",
    )



    args = parser.parse_args()
    return args