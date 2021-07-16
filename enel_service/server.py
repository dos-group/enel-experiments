from fastapi import FastAPI


def start_server():
    from modeling.routes import prediction_router, training_router
    from modeling.routes import prediction_metadata, training_metadata
    from submission.routes import router as submission_router
    from submission.routes import metadata as submission_metadata
    from common.exception_handler import register_exception_handler

    my_app = FastAPI(
        title="DRMS",
        description="The open source DRMS (Deep Resource Management System) project.",
        version="1.0.0",
        openapi_tags=[prediction_metadata, training_metadata, submission_metadata]
    )
    # routes
    my_app.include_router(prediction_router)
    my_app.include_router(training_router)
    my_app.include_router(submission_router)
    # exception handlers
    register_exception_handler(my_app)
    # return app
    return my_app


app = start_server()
