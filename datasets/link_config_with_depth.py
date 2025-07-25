link_config = {
    "config": {
        "log_dir": "./serve_logs",
        "pipe_name_file_dependencies": [
            "ds_config.py",
            "links.py",
            "transforms.py"
        ],
    },
    "links": [
        {
            "name": "nid",
            "class": "links_with_depth.NidLink",
            "parallelism": 2,
            "cpu": 1,
            "memory": 64 * 1024,
            "grouping": "none",
            "fragment_size": 256,
            "output_buffer_size": 256,
            "recieve_buffer_size": 64,
        },
        {
            "name": "train",
            "class": "links_with_depth.ProcessLink",
            "parallelism": 96,
            "cpu": 1,
            "preemptible_flag": 1,
            "memory": 3 * 1024,
            "grouping": "cwd",
            "fragment_size": 1,
            "output_buffer_size": 32,
            "recieve_buffer_size": 256,
        },
    ],
}