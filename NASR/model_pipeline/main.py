from model_pipeline.core import LatencyPredictModelPipeline


def main():
    """
        process_type: only support for "production", "research"
    """
    model_pipeline = LatencyPredictModelPipeline()
    model_pipeline.process(process_type="search")


if __name__ == '__main__':
    main()
