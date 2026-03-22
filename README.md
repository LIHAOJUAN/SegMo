# SegMo
SegMo: Algorithm-System Co-Design for efficient VideoLLM inference. By employing an Upstream Intervention strategy, SegMo unifies Content-Aware Sparsification (CAS) and Locally-Cohesive Segment Parallelism (LSP) to achieve simultaneous gains in both accuracy (+12.00%) and prefill speed (3.55x).

 #🚀**SegMo: Co-Designing Content-Aware Sparsity and Locally-Cohesive Segment Parallelism for Efficient VLM Inference**
SegMo is an end-to-end inference framework specifically designed for long-video understanding with Video Large Language Models (VideoLLM). SegMo rejects traditional downstream pruning in favor of an Upstream Intervention strategy, optimizing both what to compute and how to compute it before vision encoding. By unifying Content-Aware Sparsification (CAS) and Locally-Cohesive Segment Parallelism (LSP) under this strategy, SegMo achieves simultaneous gains in both accuracy (+12.00%) and prefill speed (3.55x).

##🏗️ **Algorithm-System Co-Design**
SegMo's efficiency stems from the synergy between its algorithmic sparsification and its system-level parallelism.
###**Content-Aware Sparsification (CAS)**: 
####"What to compute": A hierarchical algorithm that uses Query Relevance to dynamically allocate the frame budget across different scenes and Temporal Redundancy to eliminate intra-scene static redundancy. These factors are combined to proportionally allocate the frame budget ($m_k$) to each scene, ensuring computational resources are focused on information-rich, dynamic, and relevant content.
####Performance Ceiling: Unlike lossy pruning, CAS not only preserves critical semantics but can even surpass uncompressed baseline accuracy by selecting more "correct" frames.

###**Locally-Cohesive Segment Parallelism (LSP)**: 
####"How to compute": Based on the empirical discovery of Local Cohesion in VideoLLM attention , SegMo partitions the workload at scene boundaries according to real-time hardware capacity. This enables communication-free parallel prefill across multiple GPUs.
####Global Awareness: To prevent context loss, a **Global Context Injection (GCI)** mechanism prepends a lightweight map of head-frames to each parallel shard, maintaining global reasoning at near-zero cost.

##⏱️ **System & Pipeline Optimization**
###Macro-Pipeline: Overlaps the CPU preprocessing of the next request with the GPU computation of the current request to eliminate hardware idling.
###Micro-Overhead Hiding: Asynchronously parallelizes scene detection with Video I/O, ensuring detection costs are effectively masked, introducing near-zero additional latency to the end-to-end pipeline.

##Results: Simultaneous Gains in Accuracy and Latency
Validated across LVBench, LongVideoBench, and Video-MME with models like Qwen2-VL and MiniCPM-V 2.6:
###Acceleration: Achieves up to 3.55x prefill speedup under 2-GPU parallelism.
###Accuracy: Improves accuracy by up to 12.00% over the baseline.

##📝 **Citation**
```bibtex
@inproceedings{segmo2026,
  title={SegMo: Co-Designing Content-Aware Sparsity and Locally-Cohesive Segment Parallelism for Efficient VLM Inference},
  author={Li, Haojuan and Tang, Ruohan and Cheng, Dongzhou and Zhang, Zongpu and Li, Jian and Wang, Jiaqi},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
