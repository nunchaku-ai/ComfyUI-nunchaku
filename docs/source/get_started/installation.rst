Installation
============

We provide tutorial videos to help you install and use Nunchaku on Windows,
available in both `English <nunchaku_windows_tutorial_en_>`_ and `Chinese <nunchaku_windows_tutorial_zh_>`_.
If you run into issues, these resources are a good place to start.

Choose one of the following installation methods:

Option 1: Use as ComfyUI Plugin (Recommended)
---------------------------------------------

This is the standard way to use Nunchaku with your existing or new ComfyUI installation.
You can install the ComfyUI-nunchaku plugin using one of the following methods, then install the Nunchaku backend.

Step 1: Install the ComfyUI-nunchaku Plugin
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Choose one of the following methods to install the `ComfyUI-nunchaku <github_comfyui-nunchaku_>`_ plugin:

**Method A: Comfy-CLI**

You can easily use `comfy-cli <github_comfy-cli_>`_ to run ComfyUI with Nunchaku:

.. code-block:: shell

   pip install comfy-cli  # Install ComfyUI CLI
   comfy install          # Install ComfyUI (skip if already installed)
   comfy node registry-install ComfyUI-nunchaku  # Install Nunchaku

**Method B: ComfyUI-Manager**

1. Install `ComfyUI <github_comfyui_>`_ with

   .. code-block:: shell

      git clone https://github.com/comfyanonymous/ComfyUI.git
      cd ComfyUI
      pip install -r requirements.txt

2. Install `ComfyUI-Manager <github_comfyui-manager_>`_ with the following commands:

   .. code-block:: shell

      cd custom_nodes
      git clone https://github.com/ltdrdata/ComfyUI-Manager comfyui-manager

3. Launch ComfyUI

   .. code-block:: shell

      cd ..  # Return to the ComfyUI root directory
      python main.py

4. Open the Manager, search ``ComfyUI-nunchaku`` in the Custom Nodes Manager and then install it.

**Method C: Manual Installation**

1. Set up `ComfyUI <github_comfyui_>`_ with the following commands:

   .. code-block:: shell

      git clone https://github.com/comfyanonymous/ComfyUI.git
      cd ComfyUI
      pip install -r requirements.txt

2. Clone this repository into the ``custom_nodes`` directory inside ComfyUI:

   .. code-block:: shell

      cd custom_nodes
      git clone https://github.com/mit-han-lab/ComfyUI-nunchaku nunchaku_nodes

.. _install-nunchaku-backend:

Step 2: Install the Nunchaku Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Starting from **ComfyUI-nunchaku v0.3.2**,
you can easily install or update the `Nunchaku <github_nunchaku_>`_ wheel using :ref:`install-wheel-json`, once all dependencies are installed.

Alternatively, follow :ref:`nunchaku:installation-installation` to manually install the backend (pre-built wheels or build from source).

Option 2: Nunchaku Pre-installed ComfyUI Desktop (Windows Only, Experimental for Now)
-------------------------------------------------------------------------------------

Starting from **v1.0.2**, we provide a packaged version of ComfyUI that comes with ComfyUI-Manager and ComfyUI-Nunchaku built-in,
eliminating the need to download any additional dependencies. This is the easiest way to get started on Windows:

1. Download the packaged ComfyUI zip file from our `GitHub releases <https://github.com/nunchaku-tech/ComfyUI-nunchaku/releases>`__. Choose the appropriate PyTorch version for your system (e.g., torch2.7, torch2.8, or torch2.9).

2. Extract the zip file to your working directory (we recommend extracting it under ``C:\Program Files`` on Windows).

3. Navigate to the extracted folder and execute ``ComfyUI.exe`` to launch the application and complete
   the installation steps. If Windows SmartScreen blocks the execution, click "Run anyway" to proceed.

Once installed, you will be able to run the ComfyUI application directly from your PC without any
additional setup required.

Option 3: ComfyUI LTS Installation
----------------------------------

`ComfyUI LTS <https://github.com/hiddenswitch/ComfyUI>`__ is a version of ComfyUI that is installable with modern Python packaging tools like `uv <https://github.com/astral-sh/uv>`_. This method is recommended for developers.

These instructions are adapted from the `ComfyUI LTS README <https://github.com/hiddenswitch/ComfyUI#installing>`__. Please refer to it for more detailed instructions, especially for Windows.

Step 1: Install ComfyUI LTS and ComfyUI-nunchaku
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1.  Install ``uv``, a fast Python package installer. You can install it using the following command:

    .. code-block:: shell

       pip install uv

    Alternatively, if you prefer to install it system-wide or want the latest version, see the instructions at the `uv GitHub repository <https://github.com/astral-sh/uv#installation>`__.

    For example, with Homebrew on macOS:

    .. code-block:: shell

       brew install uv

    Or to install the pre-built binary (recommended for speed):

    .. code-block:: shell

       curl -LsSf https://astral.sh/uv/install.sh | sh


2.  Create a directory for your ComfyUI workspace and create a virtual environment inside it.

    .. code-block:: shell

       mkdir ComfyUI-Workspace
       cd ComfyUI-Workspace
       uv venv

3.  Install Nunchaku.

    .. code-block:: shell

       uv pip install "nunchaku@git+https://github.com/mit-han-lab/ComfyUI-nunchaku.git"

    Then manually install the Nunchaku backend wheel following :ref:`install-nunchaku-backend`.

    Alternatively, you can specify your PyTorch version as an extra to install both the plugin and backend wheel in one step. For example, for PyTorch 2.8:

    .. code-block:: shell

       uv pip install "nunchaku[torch28]@git+https://github.com/mit-han-lab/ComfyUI-nunchaku.git"

To run ComfyUI, execute the following from your workspace directory:

.. code-block:: shell

   uv run comfyui

Option 4: Build Your Own Portable Version (Windows Only)
--------------------------------------------------------

We support Windows Portable for an easy Nunchaku build.


**Build Steps**

1. Clone or download the ``portable_windows`` folder from the repository.

2. Open Command Prompt and navigate to your desired working directory:

   .. code-block:: shell

      cd C:\your\working\directory

3. Run the build script:

   .. code-block:: shell

      path\to\portable_windows\build_portable.cmd

4. Follow the interactive prompts to select:

   - **CUDA version**: Choose between CUDA 12.6 (cu126) or CUDA 12.8 (cu128, recommended)
   - **Nunchaku version**: Choose from available versions (v1.0.2 stable, v1.1.0, or v1.2.0 latest)

5. Once complete, the portable build will be created in ``ComfyUI_nunchaku_portable`` folder.

**Running the Portable Build**

Navigate to the ``ComfyUI_nunchaku_portable`` folder and run:

.. code-block:: shell

   run_nvidia_gpu.bat

.. note::

   If ComfyUI fails to start, try updating your NVIDIA drivers to the latest version.
   If you encounter a ``c10.dll`` error, install the Visual C++ Redistributable from
   `Microsoft <https://aka.ms/vs/17/release/vc_redist.x64.exe>`_.
