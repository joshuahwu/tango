# Packaging

Before we get started, note that, like many topics touched on in this repo, packaging, building, and deploying are big topics. I recommend looking at the following resources for more info:

- Read the [python packaging guide](https://pypackaging-native.github.io/) for a high-level overview of this topic.
- Read [pyOpenSci's packaging guide](https://www.pyopensci.org/python-package-guide/package-structure-code/intro.html) for information about the possible choices you can make. We recommend one way of handling these choices, this guide provides information about the other possibilities.
- Read the [scientific python development guide's](https://learn.scientific-python.org/development/tutorials/packaging/) tutorial and [guide](https://learn.scientific-python.org/development/guides/packaging-simple/) on this topic.
- Refer to [PyPA's packaging overview](https://packaging.python.org/en/latest/overview/) for the most authorative (and detailed) information about this topic. [The "packaging and distributing projects" page](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/), in particular, covers a lot of details you may want to refer to.

## Background

The primary goal is to end up with an installation that can be completed with a single `pip install` command. If you're reading this guide, your end users are probably scientists, and they want to be able to install something and have it *just work*. So providing them with a simple installation procedure is critical: if users can't install your package, they won't use it. Unfortunately, as in so much of life,  something that appears simple to the end user is generally not simple for the creator. Here, we do our best to provide a minimal working example that will cover most use-cases for writers of high-level scientific code.

Before we get started, it's important to understand that there are two types of files you can distribute via `pip`: source distribution (in python, often referred to as "sdist" and normally stored as `.tar.gz` archives) and binaries (in python, these are generally "wheel files", zip-format archives with extension `.whl`). Typically, you'll distribute both.

- Source distribution are the raw files, as stored on GitHub (or whatever platform you use), and will require the user's computer to build the package itself. For pure python packages, this is probably not a problem, as the user will need to have python anyway. However, if you have non-python compiled dependencies (called "native dependencies" for some reason, which is how I'll refer to them in the rest of this note), this will get more complicated. For example, if you depend on C or C++ code, the user will need to have a C compiler, which Windows machines do not have by default.
- Binary files are the already built / compiled source files. They include everything necessary and are ready to be installed directly. They're thus faster to install and less likely to run into user- or operating system-specific issues. However, this means that the developer must build and upload to PyPI separate wheels for each operating system they wish to support. However, if you have no native dependencies, you can produce a "universal" or ["pure python" wheel](https://packaging.python.org/en/latest/guides/distributing-packages-using-setuptools/#pure-python-wheels) (so-called because all of your code is in python), which will work on all operating systems.

In this note, we'll discuss how to build and upload your library using a GitHub action that handles as much of this as possible for you: it will build the source distribution and wheel files, check that installation is possible, run some tests, and upload to PyPI.

## Specifying project metadata and build instructions with pyproject.toml 

We follow the advice of [pyopensci](https://www.pyopensci.org/python-package-guide/package-structure-code/pyproject-toml-python-package-metadata.html) and use `pyproject.toml` to specify build requirements and metadata (rather than `setup.py`), a template for which is included in this repo. Metadata includes the authors, a brief description, homepage url, etc. which will all be rendered in the PyPI sidebar, for example. Build requirements include, at a minimum, the dependencies, and potentially other installation instructions to pass to `pip`.
Here's a more in-depth look at the configurations featured in our `pyproject.toml` template.

### 1. `[build-system]` and versioning

`[build-system]` defines the build system requirements for the project, specifying the dependencies and build backend used for building and packaging. You can provide the following details:

  - `requires`: A list of required dependencies for the build system.

  - `build-backend`: The build backend.

   **We recommend the following:**
    
   - [setuptools](https://setuptools.pypa.io/en/latest/): It is a fully-featured, widely-used library in Python that facilitates packaging, building, and installation of Python projects. It includes capabilities for dependency management, version management, and easy distribution of Python packages.
   - [setuptools-scm](https://pypi.org/project/setuptools-scm/): This plugin for setuptools handles managing your Python package versions using source control management (scm) metadata. It automatically derives the version of your package from the state of your version control system, such as Git or Mercurial.
   - [setuptools.build_meta](https://setuptools.pypa.io/en/latest/build_meta.html): This `setuptools` module allows you to define and configure the build system for your Python project.

   **Syntax example:**
   

   ```toml
   [build-system]
   requires = ["setuptools", "setuptools-scm[toml]"]  
   build-backend = "setuptools.build_meta"
   
   [project]
   dynamic = ["version"]

   [tool.setuptools_scm]
   write_to = "src/ccn_template/version.py"
   version_scheme = 'python-simplified-semver'
   local_scheme = 'no-local-version'
   ```

   **Additional Information on `setuptools-scm`:**

   How it works:

   - It examines the source code repository to determine the current version using scm metadata.
   - It simplifies versioning and release management by dynamically extracting version information from the scm system.
   - During package distribution, setuptools-scm retrieves the version number and incorporates it into the built package.
   
   What is it good for:
   
   - By using `setuptools-scm`, you can automate the versioning process and ensure accurate versioning based on the scm history. This eliminates the need for manual version specification in your project's configuration files. It will be automatically used by setuptools for managing versioning during package distribution.

   **Alternatives:**

   - [poetry](https://python-poetry.org/) or [flit](https://pypi.org/project/flit/)

!!! info "Where does the version number come from?"
    When using `setuptools-scm`, as above, the version number will be automatically determined at build-time and written to the specified file: `src/ccn_template/version.py`. We have included the line `from .version import __version__` in `src/ccn_template/__init__.py`, so that the version number can be accessed at `ccn_template.__version__`, as is typical for python. `setuptools-scm` automatically determines the version number based on the value of `version_scheme` and `local_scheme`. With the options specified in the example above (`'python-simplified-semver'` and `'no-local-version'`, respectively), if the commit being built from has a semantic version tag (see [workflow notes](./00-workflow.md#versioning)) and no changes, `setuptools-scm` will use the tag as the version number. If there have been changes, it will increment the patch version (the last number) and append `.devN`, where `N` is the number of commits since the last tag. See [setuptools-scm README](https://github.com/pypa/setuptools_scm/) for more details and, when installed, run `python -m setuptools_scm` to see what version number it determines for your package.
   
### 2. `[project.optional-dependencies]`

`[project.optional-dependencies]` specifies the optional dependencies for documentation and testing/linting. Each of these is a python list and specify optional bundles of dependencies that are installable with bracket syntax: `pip install ccn-template[dependency_bundle]` (or `pip install .[dependency_bundle]` if we're installing from a local copy). See [docs](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/#dependencies-optional-dependencies) for more info about how to specify dependencies.

   **We recommend the following:**

   - `docs`: Documentation dependencies (see the [documentation](03-documentation.md) note for additional details).
     - [mkdocstrings[python]](https://mkdocstrings.github.io/): A MkDocs plugin that generates documentation from docstrings in Python code. It extracts information from docstrings of functions, classes, modules, and other objects.
     - [mkdocs_section_index](https://oprypin.github.io/mkdocs-section-index/): A MkDocs plugin that adds a navigation section index to the documentation sidebar. It enhances navigation by providing an index of the sections in the documentation.
     - [mkdocs_gen_files](https://oprypin.github.io/mkdocs-gen-files/): A MkDocs plugin that allows the generation of additional files during the documentation build process. It provides functionality to dynamically generate files that can be included in the documentation.
     - [mkdocs_literate_nav](https://oprypin.github.io/mkdocs-literate-nav/): A MkDocs plugin that enhances the navigation sidebar by providing collapsible sections. It improves the readability and organization of the documentation by allowing users to collapse and expand sections.
     - [mkdocs-gallery](https://smarie.github.io/mkdocs-gallery/): A MkDocs plugin that enables the creation of example galleries in the documentation. These examples are written as python scripts that are converted to jupyter notebooks when the docs are built. This combines the advantages of plain scripts (easy to version control and review) and notebooks (easy for users to run and experiment with).

   - `dev`: Developer dependencies. These include the testing framework and linters, see [the testing notes](05-linters-and-tests.md) for more info.
     - [black](https://black.readthedocs.io/en/latest/): Code formatter. It automatically formats your Python code according to the Black code style.
     - [isort](https://isort.readthedocs.io/en/latest/): Import sorter. It automatically organizes and sorts import statements in your Python code.
     and maintain requirements.txt or pipfile.lock files.
     - [pytest](https://docs.pytest.org/en/7.3.x/): Testing framework. It is a popular testing framework for Python that allows you to write and execute tests easily.
     - [flake8](https://flake8.pycqa.org/en/latest/): Code linter. It checks your Python code for style, potential errors, and adherence to coding conventions.
     - [pytest-cov](https://pytest-cov.readthedocs.io/en/latest/): Test coverage plugin for pytest. It integrates test coverage measurement into the pytest framework, providing coverage reports and analysis for your tests.
     - [pydocstyle](http://www.pydocstyle.org/en/latest/index.html): Docstrings linter. It checks that the docstrings in your Python code adhere to a specified convention. This is critical for maintaining a consistent style across your project's documentation, as well as ensuring that tools which automatically generate documentation from docstrings function correctly, see the [documentation note](03-documentation.md#generating-reference-documentation-from-docstrings-).     
  
  **Syntax example:**
  
```toml
[project.optional-dependencies]
docs = [
 'mkdocs',
 'mkdocstrings[python]',
 # ...
]
dev = [
 "black", 
 "isort",                       
 # ...
]
```

### 3. `[tool.pytest.ini_options]` 

`[tool.pytest.ini_options]`: The additional command-line options for pytest. These options will be added by default when running `pytest`. You can provide the following details:

   - `addopts`: It defines which options are passed to pytest, run `pytest --help` for a complete list of available options.
   - `testpaths`: It specifies the directories to search for tests.
   
  **Syntax Example:**
```toml
[tool.pytest.ini_options]
addopts = "--cov=ccn_template"
testpaths = ["tests"]
```

  **Additional Information:**
  
  See the dedicated `pytest` [documentation](https://docs.pytest.org/en/latest/reference/reference.html#ini-options-ref) for detailed informations.


### 4. `[tool.pydocstyle]`

`[tool.pydocstyle]`: The additional command-line options for pydocstyle. These options will be added by default when running `pydocstyle`. We recommend to set:

   - `convention`: It defines which conventions should be followed by the docstrings. Options include [numpydoc](https://github.com/numpy/numpydoc), [google](https://google.github.io/styleguide/pyguide.html), and [pep257](http://www.python.org/dev/peps/pep-0257/).

**Syntax Example:**
```toml
[tool.pydocstyle]
convention = "numpy"                # Convention for linting (numpy, google, pep257)
```
   
## Manually building and deploying

In this template repo, we use python's `setuptools` to handle the building and deployment of the package. There are many other choices, such as [Poetry](https://python-poetry.org/), but `setuptools` is sufficient for our purposes here, and I find the others confusing. You can look at [PyOpenSci's packaging guide](https://www.pyopensci.org/python-package-guide/package-structure-code/python-package-build-tools.html#) for some more info on the other possibilities (note, however, that I'm not sure how they interact with `cibuildwheel`, if you'll be using that option, which we discuss more below).

Building and deployment are two separate steps. Building creates the source distribution and wheel files described earlier in this note, while deployment puts those files onto PyPI so that users can easily use them to install your package. Typically, they're performed together (there's not really a reason to build if you're not going to deploy).

This section discusses how you would manually build and deploy your package, which is useful for understanding what's going on, but in practice, we handle the building and deployment of our package with Github actions, which take advantage of the fact that Github Action is a PyPI [trusted publisher](https://docs.pypi.org/trusted-publishers/) and thus don't require us to set up API tokens or use username / password for authentication. You can do all the following manually from your local machine, but deploying to PyPI or Test PyPI will require using one of those two other methods for authentication (building does not).

!!! warning

    Note that once you deploy to PyPI, **you cannot undo it.** PyPI can only have one package with a given name and only one upload per version (i.e., you cannot overwrite or re-upload a package with the same version). Therefore, you should be careful about deploying to PyPI and test using Test PyPI first.

### Build

#### Pure python

Assuming your `pyproject.toml` is appropriately set up, you can build your package with the following lines:

```bash
pip install build # only run the first time
python -m build --outdir dist/ --sdist --wheel
```

This will build the source distribution and wheel for your system. If your code is pure python, as discussed above, then you're done! You can double check this by running `ls dist/*.whl`; if your package is pure python, the name will look like `package-version-py3-none-any.whl` (replacing `package` and `version` with the name and version, respectively, of your package). Otherwise, you have native dependencies.

#### CI Build Wheel

If, however, you have native dependencies, you'll need to build platform-specific wheels. To start, run the same lines as in the pure python example:

```bash
pip install build # only run the first time
python -m build --outdir dist/ --sdist --wheel
```

Now, if you check the contents of your `dist/` directory, you'll see that the name of your built wheel file will look like `package-version-py3-macosx_11_0_arm64.whl` or `package-version-py3-win_amd64.whl` or something similar. Basically, the built wheel will now also specify the operating system and architecture of your machine, and users will only be able to install wheels built with the same OS and architecture! This is a bit of a problem, as most of us don't have many computers running around to run python builds on.

Fortunately, there's a solution called `cibuildwheel`, which makes use of CI/CD systems (such as Github actions) to build wheels for multiple systems. You cannot build the wheel for multiple OSs locally, but you can use `cibuildwheel` to build the Linux wheel locally (regardless of your OS) to get a sense for how it works. To do so, see [the cibuildwheel documentation](https://cibuildwheel.readthedocs.io/en/stable/setup/#local) for how to install and run it. Note that you'll need [docker](https://www.docker.com/products/docker-desktop) installed.

As mentioned above, if your package has no native extensions (like this `ccn-template` repo!), then you don't need a platform-specific wheel, only a pure python wheel. In this case, `cibuildwheel` will actually raise an exception (see [this issue](https://github.com/pypa/cibuildwheel/issues/255) for the initial idea, and [this issue](https://github.com/pypa/cibuildwheel/issues/1021) for a longer discussion), and locally you can just rely on the `python -m build --wheel` command above.

### Deploy

Regardless of whether you have pure python or platform-specific wheels, the deployment procedure is the same. After building, we deploy our projects to the [python packaging index](https://pypi.org/) so they can be installed easily with `pip`.

To manually upload your package to PyPI (or Test PyPI), you would use [twine](https://twine.readthedocs.io/en/stable/) after building the source distribution and wheel files: `twine upload dist/*` for PyPI, `twine upload -r testpypi dist/*` for Test PyPI. However, we *strongly* recommend only using the Github action to upload to PyPI: once you upload a package with a specific version number to PyPI, you cannot overwrite it or re-upload the same package with the same version (you can "yank" a specific version, warning users about a version you don't want them to install). Therefore, we recommend being very careful around this.

!!! warning

    Make sure you have all the files you wish to upload to PyPI together before doing this! As noted above, you cannot overwrite or undo a deployment to PyPI or Test PyPI, so you'll be uploading all the files together at the same time.

## Github Actions

We deploy to PyPI every release, and we follow [semantic versioning](https://semver.org/) for release labeling (see [workflow notes](00-workflow.md) for more details on when to release and how this labeling works; this repo's `CONTRIBUTING.md` document also includes some suggested language for explaining the procedure to ccontributors). We do this by making use of continuous integration (CI) on github actions. See the [CI](06-ci.md) notes for more on CI and Github actions in general; in the following sections we discuss specifically how to use our `deploy` actions.

We provide two different Github actions for handling building and deployment: `deploy-pure-python.yml` and `deploy-cibw.yml` (`cibw` = "CI build wheel"). If your package is pure python, as described earlier in this note, use `deploy-pure-python.yml`, otherwise use `deploy-cibw.yml`. If you're not sure, manually build your package as described [earlier](#build) (`python -m build --wheel`) and then `ls dist/*.whl` to examine your built wheels. If your output looks like `package-version-py3-none-any.whl`, it's pure python. If it includes the name of your OS in the filename, it is package-specific and you should use `deploy-cibw.yml`.

### Setup 

Do the following steps, regardless of whether you're using the pure python or CI build wheel action.

1. Create an account on both [PyPI](https://pypi.org/) and [Test PyPI](https://test.pypi.org/). 
2. Follow the steps on the [pypi docs](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/) to create a PyPI project with a trusted publisher.

    !!! note
    
        Do this for both PyPI and Test PyPI --- it's recommended you always deploy to Test PyPI at first, to make sure things look the way you want, and then switch to PyPI when you're ready.
    
    !!! warning
    
        When you set up trusted publishers, you need to specify the name of the workflow file. This means that if you change the name of the workflow file, you'll have to do this over again!

3. Copy the relevant `deploy` file to your repository.

### Understanding the action file

#### Pure python

The action defined in `.github/workflows/deploy-pure-python.yml` builds and deploys a pure python project. This section is based on [this guide](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/) from the python packaging authority and makes use of [their action](https://github.com/pypa/gh-action-pypi-publish).

This action gets triggered on every github release and does the following:

1. Build the package.
2. Check that the version number created by `setuptools_scm` matches the most recent git tag.
2. Test that the package can be installed from the wheel.
3. Run some tests (`pytest tests/`).
4. Deploy to Test PyPI.

Note that you may want to add some extra tests (e.g., run some tutorial notebooks) to the test section. To do so, modify the "Run some tests" step of the yml file.

#### CI Build Wheel

!!! warning

    This template repo is a pure python package, and so we have been unable to test the included `deploy-cibw.yml` action. It's based on CI Build Wheel's official example, so it's probably fine, but just so you know.

The action defined in `deploy-cibw.yml` builds and deploys a project with additional non-python components. This section makes use of [cibuildwheel](https://cibuildwheel.readthedocs.io/en/stable/), which runs on GitHub Actions (or other CI providers) to build and test wheels across all platforms. The file is based on their [example github deployment](https://github.com/pypa/cibuildwheel/blob/main/examples/github-deploy.yml).

This action gets triggered on every github release and does the following on Ubuntu 22.04, Windows 2022, and MacOS 11 (see [github actions docs](https://docs.github.com/en/actions/using-github-hosted-runners/about-github-hosted-runners#supported-runners-and-hardware-resources) for details on the OS of the runners):

1. Build the package.
2. Test that the package can be installed from the wheel.
2. Check that the version number created by `setuptools_scm` matches the most recent git tag.
3. Run some tests (`pytest tests/`). We configure this using environmental variables in the `deploy-cibw.yml` file, but you can also configure it using `pyproject.toml`, which is probably preferred if you're using cibuildwheel all the time (see [cibw docs](https://cibuildwheel.readthedocs.io/en/stable/options/#configuration-file)).
4. Deploy to Test PyPI.

Note that you may want to add some extra tests (e.g., run some tutorial notebooks) to the test section. To do so, modify the `CIBW_TEST_COMMAND` line in the yml file (you may also need to modify `CIBW_TEST_REQUIRES`).

### Why is publish a separate job?

In both of these actions, `publish` is a separate job, rather than a step within the build job, and you may be wondering why. The main reason is because you might be running the tests multiple times in parallel (e.g., in `deploy-cibw.yml`, we build the wheel separately for each OS), but we only want to upload to PyPI once, including all files. To do this, we make use of Github's [upload and download artifact](https://docs.github.com/en/actions/using-workflows/storing-workflow-data-as-artifacts) actions, which allows us to share files between jobs. For the pure python build, the `build` action is only run on one OS, but using the artifact actions allows you to download the built package for examination on your local machine, if you'd like.

### When you're ready to deploy

As written, these files deploy to Test PyPI. When you're sure they do what you want, modify the file to publish to PyPI instead. To do so, remove the last two lines (`with:` and `repository-url:`) to deploy to PyPI instead of Test PyPI (I also recommend removing "test" from the name of the last step).

Additionally, these actions only trigger on release. You may want to modify the action so the workflow can be manually triggered, for easier testing. In that case, it's recommended you still only publish to PyPi / Test PyPI when a release is published, which you can handle with an `if` statement. See the [scientific python development guide](https://learn.scientific-python.org/development/guides/gha-pure/) for an example.
