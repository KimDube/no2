#!/usr/bin/env python
from ecmwfapi import ECMWFDataServer
server = ECMWFDataServer()
server.retrieve({
    "class": "ei",
    "dataset": "interim",
    "date": "20050101/20050201/20050301/20050401/20050501/20050601/20050701/20050801/20050901/20051001/20051101/20051201",
    "expver": "1",
    "grid": "0.75/0.75",
    "levtype": "sfc",
    "param": "165.128/166.128/207.128",
    "stream": "moda",
    "type": "an",
    "target": "output",
    'format': "netcdf",
})