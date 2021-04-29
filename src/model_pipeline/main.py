from model_pipeline.core import LatencyPredictModelPipeline


def main():
    """
        process_type: only support for "production", "research"

        elastic net:
            best param

            grid param
                {
                    "max_iter": [1, 5, 10],
                    "alpha": [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
                    "l1_ratio": np.arange(0.0, 1.0, 0.1)
                }

        mlp regressor:

            best param
                {'activation': 'tanh', 'alpha': 5e-05, 'hidden_layer_sizes': (512,), 'solver': 'lbfgs'}

            grid param
                {
                    "hidden_layer_sizes": [(1,), (50,), (64,), (100,), (128,), (256,), (512, )],
                    "activation": ["identity", "logistic", "tanh", "relu"], "solver": ["lbfgs", "sgd", "adam"],
                    "alpha": [0.000001, 0.00005, 0.00001, 0.0005, 0.0001]
                }
    """
    model_pipeline = LatencyPredictModelPipeline()
    model_pipeline.process(
        process_type="tuned",
        model_type="mlp",
        param={'activation': 'logistic', 'alpha': 5e-05, 'hidden_layer_sizes': (50,), 'solver': 'adam'}

    )


if __name__ == '__main__':
    main()
