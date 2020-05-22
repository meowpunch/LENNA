from model_pipeline.core import LatencyPredictModelPipeline


def main():
    """
        process_type: only support for "production", "research"
    """
    model_pipeline = LatencyPredictModelPipeline(
        bucket_name="production-bobsim",
        logger_name="food_material_price_pipeline",
        date="201908"
    )
    model_pipeline.process(
        process_type="production",
        pipe_data=False
    )


if __name__ == '__main__':
    main()
