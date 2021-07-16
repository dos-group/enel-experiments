import logging

from fastapi import HTTPException, status
from py4j.java_gateway import JavaGateway, GatewayParameters

from enel_service.common import id_generator
from enel_service.common.configuration import EllisSettings
from enel_service.common.db_schemes import ApplicationExecutionModel

ellis_settings: EllisSettings = EllisSettings.get_instance()


async def handle_ellis_initial_scale_out_prediction(document: ApplicationExecutionModel):
    try:
        gateway: JavaGateway = JavaGateway(gateway_parameters=GatewayParameters(address=ellis_settings.py4j_address,
                                                                                port=ellis_settings.py4j_port))

        best_scale_out: int = gateway.entry_point.computeInitialScaleOut(
            document.spark_template_values.get("scale_out_tuner", {}).get("config", {}).get("db_path", ""),
            int(document.spark_template_values.get("release_name", "").split("run")[-1]),  # get number of this run
            document.global_specs.algorithm_name,
            document.global_specs.min_scale_out,
            document.global_specs.max_scale_out,
            int(document.global_specs.max_runtime * 1000)  # must be in milliseconds
        )

        document.start_scale_out = best_scale_out
        document.end_scale_out = best_scale_out
    except BaseException as exc:
        logging.error("Could not communicate with Py4J-Server!", exc_info=exc)
        raise HTTPException(status_code=status.HTTP_412_PRECONDITION_FAILED,
                            detail="Could not communicate with Py4J-Server!")

    return document, f"ellis-placeholder-id-{id_generator(30)}"
