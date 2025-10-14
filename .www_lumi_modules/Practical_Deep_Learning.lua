-- Jupyter
prepend_path("MODULEPATH","/appl/local/csc/modulefiles/")
depends_on("pytorch/2.7")
setenv("_COURSE_BASE_NAME","PDL-2025-11")
-- Relative to the course dir
setenv("_COURSE_NOTEBOOK","intro-to-dl/day1/01-pytorch-test-setup.ipynb")
setenv("_COURSE_GIT_REPO","https://github.com/csc-training/intro-to-dl/")
-- Anything valid for checkout
-- setenv("_COURSE_GIT_REF","")
-- lab / notebook / empty (defaults to jupyter)
setenv("_COURSE_NOTEBOOK_TYPE","lab")
