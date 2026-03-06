"""Entry point for training and evaluating U-Net neuron segmentation models.

Usage examples:

    # Train a model with default config (see config.py)
    python main.py --train --model_name my_model

    # Train with custom hyperparameters
    python main.py --train --model_name my_model --epochs 200 --lr 0.0005 --batch_size 4

    # Evaluate a trained model on the test set
    python main.py --test --model_name my_model

    # Find optimal clustering thresholds on the validation set
    python main.py --find_th --model_name my_model
"""

import argparse

from src.training import helpers as h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="U-Net + Mean-Shift neuron segmentation pipeline"
    )

    # Mode selection
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true", help="Train a new model")
    mode.add_argument("--test", action="store_true", help="Evaluate on test set")
    mode.add_argument("--find_th", action="store_true",
                      help="Search for optimal clustering thresholds")

    # Model identification
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name for saving/loading the model")

    # Training hyperparameters (override config.py defaults)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--embedding_dim", type=int, default=32)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--scaling", type=float, default=4.0)
    parser.add_argument("--subsample_size", type=int, default=1024)

    # Mean-shift parameters
    parser.add_argument("--nb_iterations", type=int, default=0)
    parser.add_argument("--kernel_bandwidth", type=float, default=6.0)
    parser.add_argument("--step_size", type=float, default=1.0)

    # Testing thresholds
    parser.add_argument("--cl_th", type=float, default=1.5,
                        help="Clustering threshold")
    parser.add_argument("--pp_th", type=float, default=0.2,
                        help="Post-processing threshold")
    parser.add_argument("--obj_size", type=int, default=20,
                        help="Minimum object size (pixels)")
    parser.add_argument("--hole_size", type=int, default=20,
                        help="Minimum hole size to fill (pixels)")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.train:
        setup = h.Setup(
            model_name=args.model_name,
            subsample_size=args.subsample_size,
            embedding_dim=args.embedding_dim,
            margin=args.margin,
            scaling=args.scaling,
            nb_epochs=args.epochs,
            save_config=True,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            include_background=False,
            background_pred=True,
            nb_iterations=args.nb_iterations,
            kernel_bandwidth=args.kernel_bandwidth,
            step_size=args.step_size,
            embedding_loss=True,
        )
        setup.main()

    elif args.test:
        h.test(
            args.model_name,
            cl_th=args.cl_th,
            pp_th=args.pp_th,
            obj_size=args.obj_size,
            hole_size=args.hole_size,
            show_image=True,
            save_image=False,
        )

    elif args.find_th:
        h.find_th(args.model_name, iter=10)


if __name__ == "__main__":
    main()

