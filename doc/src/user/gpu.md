# select gpu

Environment variable `CUDA_VISIBLE_DEVICES` is used to select device
to run. Example,

    CUDA_VISIBLE_DEVICES=1 ./udx

For multi-node runs `UDEVICES` is used to select devices. Example

    UDEVICES=1,2 ./udx
