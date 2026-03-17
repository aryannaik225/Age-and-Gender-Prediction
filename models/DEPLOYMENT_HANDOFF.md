# Demographics Pipeline: Deployment Handoff Notes

## Why ONNX instead of TensorRT (.engine)?
Attached are the final production models exported in **ONNX (Open Neural Network Exchange)** format rather than pre-compiled TensorRT `.engine` files. 

TensorRT engines are strictly tied to the specific GPU architecture, CUDA version, and TensorRT version of the machine that compiles them. If these models were compiled into TensorRT on the development machine, they would instantly crash when loaded onto the production server hardware due to architecture mismatch.

By providing the `.onnx` files, the deployment team can compile the TensorRT engines natively on the target production GPUs, ensuring maximum hardware-specific graph optimizations and memory fusing.

## Model Details & Architecture
Both models are based on the **Swin Transformer (Tiny)** architecture, operating on `224x224` RGB face crops normalized using ImageNet standards.

* **Gender Model (`swin_gender_v2_0.onnx`):** Standard Multi-class classification.
    * Outputs 2 logits (Index 0: Female, Index 1: Male).
* **Age Model (`swin_ordinal_age_v2.onnx`):** Uses **Ordinal Regression** rather than standard classification to maintain the chronological logic of aging.
    * Outputs 4 logits representing binary thresholds (e.g., *Is age > 12? Is age > 19?*). 
    * Decoding logic: Pass outputs through a Sigmoid activation, threshold at `0.5`, and sum the `True` values to determine the predicted bucket index (0 through 4).

*Note: Dynamic axes have been enabled for `batch_size` (Dim 0) on both inputs and outputs to support multi-face concurrent processing.*

## TensorRT Compilation Instructions
To compile these ONNX models into highly optimized TensorRT engines on the target deployment server, use the standard Nvidia `trtexec` command line tool. 

Example compilation command for FP16 precision:

```bash
# Convert Gender Model
trtexec --onnx=swin_gender_v2_0.onnx --saveEngine=swin_gender_fp16.engine --fp16

# Convert Age Model
trtexec --onnx=swin_ordinal_age_v2.onnx --saveEngine=swin_age_fp16.engine --fp16
```
***(If INT8 quantization is required for extreme latency constraints, standard INT8 calibration caches will need to be generated against the staging dataset).***
