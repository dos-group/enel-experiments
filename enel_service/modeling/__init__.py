import datetime

from bson import ObjectId

request_id: str = str(ObjectId())
application_database_obj = {
    "_id": request_id,
    "application_id": "application123",
    "start_time": datetime.datetime(2020, 5, 12, 20),
    "end_time": datetime.datetime(2020, 5, 12, 21),
    "start_scale_out": 6,
    "end_scale_out": 6,
    "fit_time": 12.0,
    "predict_time": 0.01,
    "preparation_time": 5.23,
    "global_specs": {
        "solution_name": "enel",
        "system_name": "spark",
        "algorithm_name": "grep",
        "data_size_MB": "20000",
        "machine_name": "2x.large",
        "environment_name": "aws public cloud",
        "experiment_name": "default",
        "min_scale_out": 2,
        "max_scale_out": 12,
        "max_runtime": 100
    },
    "optional_specs": {},
    "master_specs": {
        "scale_out": 1,
        "cores": 6,
        "memory": "10240m",
        "memory_overhead": "2048m",
    },
    "worker_specs": {
        "scale_out": 1,
        "cores": 6,
        "memory": "10240m",
        "memory_overhead": "2048m",
    }
}

job_database_obj = {
    "_id": request_id,
    "application_execution_id": request_id,
    "application_id": "application123",
    "job_id": 1,
    "start_time": datetime.datetime(2020, 5, 12, 20),
    "end_time": datetime.datetime(2020, 5, 12, 21),
    "start_scale_out": 5,
    "end_scale_out": 8,
    "fit_time": 12.0,
    "predict_time": 0.01,
    "preparation_time": 5.23,
    "global_specs": {
        "solution_name": "enel",
        "system_name": "spark",
        "algorithm_name": "grep",
        "data_size_MB": "20000",
        "machine_name": "2x.large",
        "environment_name": "aws public cloud",
        "experiment_name": "default",
        "min_scale_out": 2,
        "max_scale_out": 12,
        "max_runtime": 100
    },
    "optional_specs": {},
    "master_specs": {
        "scale_out": 1,
        "cores": 6,
        "memory": "10240m",
        "memory_overhead": "2048m",
    },
    "worker_specs": {
        "scale_out": 1,
        "cores": 6,
        "memory": "10240m",
        "memory_overhead": "2048m",
    },
    "stages": {
        "1": {
            "stage_id": 1,
            "stage_name": "stage 1",
            "num_tasks": 10,
            "parent_stage_ids": [],
            "start_time": datetime.datetime(2020, 5, 12, 20, 0),
            "end_time": datetime.datetime(2020, 5, 12, 20, 20),
            "start_scale_out": 5,
            "end_scale_out": 6,
            "rdd_num_partitions": 10,
            "rdd_num_cached_partitions": 2,
            "rdd_mem_size": 12000,
            "rdd_disk_size": 0,
            "metrics": {
                "cpu_utilization": 0.7,
                "gc_time_ratio": 0.01,
                "shuffle_rw_ratio": 0.1,
                "data_io_ratio": 0.3,
                "memory_spill_ratio": 0.05
            },
            "rescaling_time_ratio": 0.0
        },
        "2": {
            "stage_id": 2,
            "stage_name": "stage 2",
            "num_tasks": 15,
            "parent_stage_ids": [1],
            "start_time": datetime.datetime(2020, 5, 12, 20, 20),
            "end_time": datetime.datetime(2020, 5, 12, 20, 40),
            "start_scale_out": 6,
            "end_scale_out": 7,
            "rdd_num_partitions": 15,
            "rdd_num_cached_partitions": 5,
            "rdd_mem_size": 18000,
            "rdd_disk_size": 0,
            "metrics": {
                "cpu_utilization": 0.6,
                "gc_time_ratio": 0.02,
                "shuffle_rw_ratio": 0.3,
                "data_io_ratio": 0.4,
                "memory_spill_ratio": 0.07
            },
            "rescaling_time_ratio": 0.0
        },
        "3": {
            "stage_id": 3,
            "stage_name": "stage 3",
            "num_tasks": 10,
            "parent_stage_ids": [2],
            "start_time": datetime.datetime(2020, 5, 12, 20, 40),
            "end_time": datetime.datetime(2020, 5, 12, 20, 50),
            "start_scale_out": 7,
            "end_scale_out": 8,
            "rdd_num_partitions": 20,
            "rdd_num_cached_partitions": 7,
            "rdd_mem_size": 19000,
            "rdd_disk_size": 0,
            "metrics": {
                "cpu_utilization": 0.4,
                "gc_time_ratio": 0.05,
                "shuffle_rw_ratio": 0.6,
                "data_io_ratio": 0.2,
                "memory_spill_ratio": 0.07
            },
            "rescaling_time_ratio": 0.0
        },
    }
}
