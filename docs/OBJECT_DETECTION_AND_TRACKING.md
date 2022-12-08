## Object detection and tracking 👨🏻‍💻

### Table of contents

- [Object detection](#object-detection)
  * [Via mamba](#new-env-mamba)
- [Object tracking](#object-tracking)
  * [Via conda](#new-env-conda)

#

<a id="create-new-env" />

### 1. Create new enviornment

<a id="new-env-mamba" />

#### 1.1. Via mamba

```bash
brew install mambaforge
```

And then later install `mamba` via

```bash
conda install mamba -n base -c conda-forge
```

Now, you can create a new env via

```bash
mamba env create -n visual_analysis -f docs/config/visual_analysis_env.yaml
```

<a id="new-env-conda" />

#### 1.2. Via conda

Before starting further, make sure that you have `conda` (Anaconda) installed (otherwise, create a new env via [virutalenv](#new-env-virtualenv)). We will create a new enviornment for the purpose of our labs:

```bash
conda create -n visual_analysis -f docs/config/visual_analysis_env.yaml -y
```

and activate it

```bash
conda activate visual_analysis
```

<a id="new-env-virtualenv" />

#### 1.3. Via virtualenv

You can create your virtual enviornment without conda as well. In order to do that, make sure that you have [`virtualenv`](https://pypi.org/project/virtualenv/) installed or else, you can install it via:


```bash
pip install virtualenv
```

Now, create your new enviornment called `visual_analysis`

```bash
virtualenv -p python3 visual_analysis
```

and then activate it via

```bash
source visual_analysis/bin/activate
```