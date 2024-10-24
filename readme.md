Requires OpenCV installation.

```bash
mkdir build
pushd build
cmake ..
cmake --build .
popd
./build/cvtf-demo
```

Place any image you want to use in `assets/images/` and modify `main()` or possibly `config.hpp.in` accordingly.
