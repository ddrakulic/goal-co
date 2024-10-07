

def add_common_args(parser):
    # -- Network --
    parser.add_argument("--dim_node_idx", type=int, default=1, help="Dimension of node random idx")
    parser.add_argument("--dim_emb", type=int, default=128, help="Embeddings size")
    parser.add_argument("--dim_ff", type=int, default=512, help="FF size")
    parser.add_argument("--num_layers", type=int, default=9, help="Encoder layers")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of heads")
    parser.add_argument("--activation_ff", type=str, default="relu", help="ReLu or GeLu")
    parser.add_argument("--node_feature_low_dim", type=int, default=8, help="Node proto feature dimension")
    parser.add_argument("--edge_feature_low_dim", type=int, default=4, help="Edge proto feature dimension")
    parser.add_argument("--activation_edge_adapter", type=str, default="relu", help="ReLu or GeLu")
    parser.add_argument("--attention_clip_value", type=float, default=10.,
                        help="Soft clipping value for attention scores")

    # -- Data --
    parser.add_argument("--problems", nargs="+", type=str, help="List of problems for training the model"
                                                                "TSP, CVRP, CWRPTW, KP, OP...")
    parser.add_argument("--train_datasets", nargs="+", type=str, help="Training datasets")
    parser.add_argument("--val_datasets", nargs="+", type=str, help="Validation datasets")
    parser.add_argument("--test_datasets", nargs="+", type=str, help="Test datasets")
    parser.add_argument("--output_dir", type=str, default="output/", help="Output dir")

    # -- Job, logger
    parser.add_argument("--job_id", type=int, default=0, help="Job id")
    parser.add_argument("--aimstack_dir", type=str, help="Aimstack dir")
    # -- Reload --
    parser.add_argument("--pretrained_model", type=str, help="Load pretrained parameters")
    # -- Eval --
    parser.add_argument("--beam_size", type=int, default=1, help="Number of beams, =1 for greedy search")
    parser.add_argument("--knns", type=int, default=-1, help="Number of KNNs used during the decoding")
    # -- Common --
    parser.add_argument("--seed", type=int, help="Seed")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=512, help="Validation batch sizes")
    parser.add_argument("--test_batch_size", type=int, default=512, help="Test batch sizes")
    parser.add_argument("--test_datasets_size", type=int, default=None, help="Size of test datasets")
    parser.add_argument("--debug", dest="debug", action="store_true")


def add_common_training_args(parser):
    # -- Data --
    parser.add_argument("--train_datasets_size", type=int, default=1000000, help="Size of training datasets")
    parser.add_argument("--val_datasets_size", type=int, default=None, help="Size of test datasets")

    # -- Optim --
    parser.add_argument("--test_every", type=int, default=50, help="Test every n epochs")
    parser.add_argument("--val_every", type=int, default=1, help="Validate every n epochs")
    parser.add_argument("--num_total_epochs", type=int, default=500, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--update_lr_every_n_epoch", type=int, default=10, help="Update lr after N epochs")
    parser.add_argument("--lr_gamma", type=float, default=0.97, help="Decay rate for scheduler")
    # -- Experiment --
    parser.add_argument("--backends", type=str, default="10000",
                        help="where to log metrics, str of bool, length 4 for stdout, tensorboard, wandb, mlflow, aim")
    parser.add_argument("--project_name", type=str, default=None, help="Project name, for logger trackers")

def add_common_self_supervised_tuning_args(parser):
    # -- Data --
    parser.add_argument("--num_samples", type=int, default=128, help="Number of samples per instance")
    parser.add_argument("--problem_size", type=int, default=100, help="Problem size for self supervised tuning")
    parser.add_argument("--num_sampled_init_params", type=int, default=16, help="Number of sampled starting params")
