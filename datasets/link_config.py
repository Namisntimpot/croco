link_config = {
    "config": {
        "log_dir": "./serve_logs",
        "pipe_name_file_dependencies": [
            "datasets/ds_config.py",
            "datasets/links.py",
            "datasets/transforms.py"
        ],
    },
    "links": [
        {
            "name": "nid",
            "class": "datasets.links.NidLink",
            "parallelism": 2,
            "cpu": 1,
            "memory": 32 * 1024,
            "grouping": "none",
            "fragment_size": 256,
            "output_buffer_size": 512,
            "recieve_buffer_size": 512,
        },
        {
            "name": "train",
            "class": "datasets.links.ProcessLink",
            "parallelism": 64,
            "cpu": 1,
            "preemptible_flag": 1,
            "memory": 3 * 1024,
            "grouping": "cwd",
            "fragment_size": 4,
            "output_buffer_size": 64,
            "recieve_buffer_size": 256,
        },
    ],
}