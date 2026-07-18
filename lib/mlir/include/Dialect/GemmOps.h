#pragma once
// TODO: include TableGen 生成的 Op 声明头文件
#include "mlir/IR/Types.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Bytecode/BytecodeOpInterface.h" 
#include "mlir/IR/BuiltinTypes.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/GemmTypes.h.inc"  
#define GET_ATTRDEF_CLASSES
#include "Dialect/GemmAttrs.h.inc"  

#define GET_OP_CLASSES
#include "Dialect/GemmOps.h.inc"