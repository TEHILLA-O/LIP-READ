"""Visual Speech Recognition ML package.

Layout:
    preprocessing/  video -> aligned mouth ROI -> tensor
    models/         model adapters behind one interface
    decoding/       tokens -> text, with confidence
    inference/      orchestration and the structured result
    streaming/      rolling buffers, utterance segmentation, inference queue
    evaluation/     WER/CER metrics
"""

__version__ = "0.1.0"
