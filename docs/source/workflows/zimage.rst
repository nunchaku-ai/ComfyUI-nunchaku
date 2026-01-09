Z-Image-Turbo
=============

.. _nunchaku-z-image-turbo-json:

nunchaku-z-image-turbo.json
---------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-z-image-turbo.png
    :alt: nunchaku-z-image-turbo.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo.json

Text-to-image workflow using the Nunchaku Z-Image-Turbo model.

**Links:**

- Workflow: :download:`nunchaku-z-image-turbo.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo.json>`
- Nunchaku Z-Image-Turbo: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-z-image-turbo>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-z-image-turbo>`
  (Place in ``models/diffusion_models``)


.. _nunchaku-z-image-turbo-lora-json:

nunchaku-z-image-turbo-lora.json
--------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-z-image-turbo-lora.png
    :alt: nunchaku-z-image-turbo-lora.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo-lora.json

Text-to-image workflow using the Nunchaku Z-Image-Turbo model, with one or multiple LoRAs loaded by native Lora Loader provided by ComfyUI.

.. note::
  - Place your LoRA files in ``models/loras``.
  - You should use LoRAs that are compatible with Z-Image-Turbo model. LoRAs for other base models (such as Flux, Qwen-Image, etc.) will NOT work with Nunchaku Z-Image-Turbo model.


**Links:**

- Workflow: :download:`nunchaku-z-image-turbo-lora.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo-lora.json>`
- Nunchak Z-Image-Turbo: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-z-image-turbo>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-z-image-turbo>`
  (Place in ``models/diffusion_models``)

.. _nunchaku-z-image-turbo-controlnet-json:

nunchaku-z-image-turbo-controlnet.json
--------------------------------------

.. image:: https://huggingface.co/datasets/nunchaku-tech/cdn/resolve/main/ComfyUI-nunchaku/workflows/nunchaku-z-image-turbo-controlnet.png
    :alt: nunchaku-z-image-turbo-controlnet.json
    :target: https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo-controlnet.json

Text-to-image workflow using the Nunchaku Z-Image-Turbo model with official controlnet (https://huggingface.co/alibaba-pai/Z-Image-Turbo-Fun-Controlnet-Union).

.. note::
  Place controlnet model file in ``models/model_patches``.


**Links:**

- Workflow: :download:`nunchaku-z-image-turbo-controlnet.json <https://github.com/nunchaku-tech/ComfyUI-nunchaku/blob/main/example_workflows/nunchaku-z-image-turbo-controlnet.json>`
- Nunchak Z-Image-Turbo: :download:`Hugging Face <https://huggingface.co/nunchaku-tech/nunchaku-z-image-turbo>` or :download:`ModelScope <https://modelscope.cn/models/nunchaku-tech/nunchaku-z-image-turbo>`
  (Place in ``models/diffusion_models``)

.. seealso::
    See nodes :ref:`nunchaku-z-image-dit-loader`.
