import sysconfig
from distutils.core import setup
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import os

# This removes the suffix with platform from output filename
# E.g. We don't want binary_classifier.cpython-310-aarch64-linux-gnu.so, we want binary_classifier.so
class NoSuffixBuilder(build_ext):
    def get_ext_filename(self, ext_name):
        filename = super().get_ext_filename(ext_name)
        suffix = sysconfig.get_config_var('EXT_SUFFIX')
        ext = os.path.splitext(filename)[1]
        return filename.replace(suffix, "") + ext

# run python3 setup_classifier.py build_ext --inplace && mv *.so ../
setup(
  name = "binary_classifer",
  ext_modules = cythonize("binary_classifier.py"),
  cmdclass={"build_ext": NoSuffixBuilder},
)


