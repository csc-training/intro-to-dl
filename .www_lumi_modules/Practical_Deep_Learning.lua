-- Jupyter
prepend_path("MODULEPATH","/appl/local/laifs/ood/lumi-multitorch")
depends_on("full-u24r70f21m50t210-20260807_115122.lua")
prepend_path("PATH", "/project/project_xxxxx/www_lumi_modules/wrappers")
setenv("_COURSE_BASE_NAME","PDL-2026-Test")
-- Relative to the course dir
setenv("_COURSE_NOTEBOOK","intro-to-dl/day1/01-pytorch-test-setup.ipynb")
setenv("_COURSE_GIT_REPO","https://github.com/csc-training/intro-to-dl/")
-- Anything valid for checkout
-- setenv("_COURSE_GIT_REF","")
-- lab / notebook / empty (defaults to jupyter)
setenv("_COURSE_NOTEBOOK_TYPE","lab")