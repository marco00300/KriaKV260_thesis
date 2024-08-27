ARCH=/opt/vitis_ai/compiler/arch/DPUCZDX8G/KV260/arch.json
      TARGET=kv260
      echo "-----------------------------------------"
      echo "COMPILING MODEL FOR KV260.."
      echo "-----------------------------------------"
      
compile() {
      vai_c_tensorflow2 \
            --model           build/quant_model/q_model4800.h5 \
            --arch            $ARCH \
            --output_dir      build/compiled_$TARGET \
            --net_name        customcnn
}


compile 2>&1 | tee build/logs/compile_$TARGET.log


echo "-----------------------------------------"
echo "MODEL COMPILED"
echo "-----------------------------------------"
