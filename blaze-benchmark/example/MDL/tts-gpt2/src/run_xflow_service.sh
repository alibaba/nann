#!/usr/bin/env bash
sudo /sbin/ldconfig
python3 /home/service/run_xflow_service.py --port=$1
