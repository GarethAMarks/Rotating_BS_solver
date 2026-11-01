#ifndef PARAMS_IO_H_
#define PARAMS_IO_H_

#include <string>
#include "bs_params.h"

BSParams read_params(const std::string& file_path);
void print_params_summary(const BSParams& p);

#endif // PARAMS_IO_H_
