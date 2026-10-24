#!/usr/bin/env bash
# ==============================================================================
# build_firmware.sh — Fast STM32F401RE Firmware Builder & Flasher
# Target: STM32 Nucleo-64 F401RE (Cortex-M4, 16 MHz HSI)
#
# Usage:
#   ./build_firmware.sh [MODE] [--flash]
#
# Examples:
#   ./build_firmware.sh 0            # Build Mode 0 (32-bit Float + FPU)
#   ./build_firmware.sh 2            # Build Mode 2 (32-bit Fixed PoT)
#   ./build_firmware.sh 5            # Build Mode 5 (16-bit PoT + SIMD SMLAD)
#   ./build_firmware.sh 5 --flash    # Build Mode 5 and flash directly to board
# ==============================================================================

set -e

MODE=${1:-0}
DO_FLASH=0
if [[ "$2" == "--flash" || "$1" == "--flash" ]]; then
    DO_FLASH=1
    if [[ "$1" == "--flash" ]]; then MODE=0; fi
fi

# Locate ARM toolchain
TOOLBIN=""
CANDIDATES=(
    "/home/abrar/st/stm32cubeide_1.18.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.13.3.rel1.linux64_1.0.0.202410170706/tools/bin"
    "/opt/st/stm32cubeide_2.1.1/plugins/com.st.stm32cube.ide.mcu.externaltools.gnu-tools-for-stm32.14.3.rel1.linux64_1.0.100.202602081740/tools/bin"
)

for c in "${CANDIDATES[@]}"; do
    if [ -f "$c/arm-none-eabi-gcc" ]; then
        TOOLBIN="$c"
        break
    fi
done

if [ -z "$TOOLBIN" ]; then
    if which arm-none-eabi-gcc >/dev/null 2>&1; then
        TOOLBIN="$(dirname "$(which arm-none-eabi-gcc)")"
    else
        echo "ERROR: arm-none-eabi-gcc not found."
        exit 1
    fi
fi

CC="$TOOLBIN/arm-none-eabi-gcc"
OBJCOPY="$TOOLBIN/arm-none-eabi-objcopy"
SIZE="$TOOLBIN/arm-none-eabi-size"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUBE_PROJ="/home/abrar/STM32CubeIDE/workspace_1.18.1/ENDPROJ_1"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# Mode names
declare -A MODE_NAMES=(
    [0]="Mode 0: 32-bit Float with Hardware FPU (FP32)"
    [1]="Mode 1: 32-bit Int Without FPU (Normal Division)"
    [2]="Mode 2: 32-bit Int Without FPU (Power-of-2 Bit-Shift)"
    [3]="Mode 3: 16-bit Float with Hardware FPU (FP16/FP32)"
    [4]="Mode 4: 16-bit Int Without FPU (Normal Division)"
    [5]="Mode 5: 16-bit Int Without FPU (PoT + ARM SMLAD SIMD Assembly)"
)

# Float ABI configuration
# Modes 0 and 3 use Hardware FPU (-mfloat-abi=hard -mfpu=fpv4-sp-d16)
# Modes 1, 2, 4, 5 test integer / without-FPU performance (-mfloat-abi=soft)
if [[ "$MODE" == "0" || "$MODE" == "3" ]]; then
    FPU_FLAGS="-mfloat-abi=hard -mfpu=fpv4-sp-d16"
else
    FPU_FLAGS="-mfloat-abi=soft"
fi

echo "======================================================================"
echo "  BUILDING: ${MODE_NAMES[$MODE]}"
echo "  Target  : STM32F401RET6 (Cortex-M4 @ 16 MHz HSI)"
echo "  ABI     : $FPU_FLAGS"
echo "======================================================================"

# Sync latest files to CubeIDE project
cp "$SCRIPT_DIR/ml_model/mlp_model.h" "$CUBE_PROJ/Core/Inc/mlp_model.h"
cp "$SCRIPT_DIR/ml_model/mlp_model.h" "$CUBE_PROJ/Core/Src/mlp_model.h"
cp "$SCRIPT_DIR/ml_model/mlp_model.c" "$CUBE_PROJ/Core/Src/mlp_model.c"
cp "$SCRIPT_DIR/main.c" "$CUBE_PROJ/Core/Src/main.c"

# Common compiler flags
CFLAGS="-mcpu=cortex-m4 -mthumb $FPU_FLAGS -O2 -Wall -Wextra -DSTM32F401xE -DBENCHMARK_MODE=$MODE"
INCLUDES="-I$CUBE_PROJ/Core/Inc -I$CUBE_PROJ/Drivers/CMSIS/Device/ST/STM32F4xx/Include -I$CUBE_PROJ/Drivers/CMSIS/Include -I$SCRIPT_DIR/ml_model"

# 1. Compile startup code
$CC -mcpu=cortex-m4 -mthumb -c "$CUBE_PROJ/Core/Startup/startup_stm32f401retx.s" -o "$BUILD_DIR/startup.o"

# 2. Compile system initialization
$CC $CFLAGS $INCLUDES -c "$CUBE_PROJ/Core/Src/system_stm32f4xx.c" -o "$BUILD_DIR/system.o"

# 3. Compile syscalls
$CC $CFLAGS $INCLUDES -c "$CUBE_PROJ/Core/Src/syscalls.c" -o "$BUILD_DIR/syscalls.o"

# 4. Compile main.c
$CC $CFLAGS $INCLUDES -c "$SCRIPT_DIR/main.c" -o "$BUILD_DIR/main.o"

# 5. Compile mlp_model.c
$CC $CFLAGS $INCLUDES -c "$SCRIPT_DIR/ml_model/mlp_model.c" -o "$BUILD_DIR/mlp_model.o"

# 6. Link ELF binary
ELF="$BUILD_DIR/firmware_mode${MODE}.elf"
BIN="$BUILD_DIR/firmware_mode${MODE}.bin"

$CC -mcpu=cortex-m4 -mthumb $FPU_FLAGS -O2 \
    -T"$CUBE_PROJ/STM32F401RETX_FLASH.ld" \
    "$BUILD_DIR/startup.o" \
    "$BUILD_DIR/system.o" \
    "$BUILD_DIR/syscalls.o" \
    "$BUILD_DIR/main.o" \
    "$BUILD_DIR/mlp_model.o" \
    -Wl,--gc-sections --specs=nano.specs -lm \
    -o "$ELF"

# 7. Generate raw binary for flashing
$OBJCOPY -O binary "$ELF" "$BIN"

echo ""
echo "  Firmware Memory Footprint:"
$SIZE "$ELF"
echo "  ✓ Successfully generated $BIN"

# 8. Flash if requested
if [ "$DO_FLASH" -eq 1 ]; then
    echo ""
    echo "======================================================================"
    echo "  Flashing to STM32 Nucleo via st-flash at 0x08000000..."
    echo "======================================================================"
    st-flash write "$BIN" 0x08000000
    echo "  ✓ Flashed successfully!"
fi
