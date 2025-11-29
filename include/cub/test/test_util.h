/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#ifdef _WIN32
#  include <windows.h>
#  undef small // Windows is terrible for polluting macro namespace
#else
#  include <sys/resource.h>
#endif

#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_math.cuh>
#include <cub/util_namespace.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/discard_iterator.h>

#include <cuda/std/__algorithm_>

#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include <nv/target>


/******************************************************************************
 * Command-line parsing functionality
 ******************************************************************************/

/**
 * Utility for parsing command line arguments
 */
struct CommandLineArgs
{
  std::vector<std::string> keys;
  std::vector<std::string> values;
  std::vector<std::string> args;
  cudaDeviceProp deviceProp;
  float device_giga_bandwidth;
  std::size_t device_free_physmem;
  std::size_t device_total_physmem;

  /**
   * Constructor
   */
  CommandLineArgs(int argc, char** argv)
      : keys(10)
      , values(10)
  {
    using namespace std;

    for (int i = 1; i < argc; i++)
    {
      string arg = argv[i];

      if ((arg[0] != '-') || (arg[1] != '-'))
      {
        args.push_back(arg);
        continue;
      }

      string::size_type pos;
      string key, val;
      if ((pos = arg.find('=')) == string::npos)
      {
        key = string(arg, 2, arg.length() - 2);
        val = "";
      }
      else
      {
        key = string(arg, 2, pos - 2);
        val = string(arg, pos + 1, arg.length() - 1);
      }

      keys.push_back(key);
      values.push_back(val);
    }
  }

  /**
   * Checks whether a flag "--<flag>" is present in the commandline
   */
  bool CheckCmdLineFlag(const char* arg_name)
  {
    using namespace std;

    for (std::size_t i = 0; i < keys.size(); ++i)
    {
      if (keys[i] == string(arg_name))
      {
        return true;
      }
    }
    return false;
  }

  /**
   * Returns number of naked (non-flag and non-key-value) commandline parameters
   */
  template <typename T>
  int NumNakedArgs()
  {
    return args.size();
  }

  /**
   * Returns the commandline parameter for a given index (not including flags)
   */
  template <typename T>
  void GetCmdLineArgument(std::size_t index, T& val)
  {
    using namespace std;
    if (index < args.size())
    {
      istringstream str_stream(args[index]);
      str_stream >> val;
    }
  }

  /**
   * Returns the value specified for a given commandline parameter --<flag>=<value>
   */
  template <typename T>
  void GetCmdLineArgument(const char* arg_name, T& val)
  {
    using namespace std;

    for (std::size_t i = 0; i < keys.size(); ++i)
    {
      if (keys[i] == string(arg_name))
      {
        istringstream str_stream(values[i]);
        str_stream >> val;
      }
    }
  }

  /**
   * Returns the values specified for a given commandline parameter --<flag>=<value>,<value>*
   */
  template <typename T>
  void GetCmdLineArguments(const char* arg_name, std::vector<T>& vals)
  {
    using namespace std;

    if (CheckCmdLineFlag(arg_name))
    {
      // Clear any default values
      vals.clear();

      // Recover from multi-value string
      for (std::size_t i = 0; i < keys.size(); ++i)
      {
        if (keys[i] == string(arg_name))
        {
          string val_string(values[i]);
          istringstream str_stream(val_string);
          string::size_type old_pos = 0;
          string::size_type new_pos = 0;

          // Iterate comma-separated values
          T val;
          while ((new_pos = val_string.find(',', old_pos)) != string::npos)
          {
            if (new_pos != old_pos)
            {
              str_stream.width(new_pos - old_pos);
              str_stream >> val;
              vals.push_back(val);
            }

            // skip over comma
            str_stream.ignore(1);
            old_pos = new_pos + 1;
          }

          // Read last value
          str_stream >> val;
          vals.push_back(val);
        }
      }
    }
  }

  /**
   * The number of pairs parsed
   */
  int ParsedArgc()
  {
    return (int) keys.size();
  }

  /**
   * Initialize device
   */
  cudaError_t DeviceInit(int dev = -1)
  {
    cudaError_t error = cudaSuccess;

    do
    {
      int deviceCount;
      error = CubDebug(cudaGetDeviceCount(&deviceCount));
      if (error)
      {
        break;
      }

      if (deviceCount == 0)
      {
        fprintf(stderr, "No devices supporting CUDA.\n");
        exit(1);
      }
      if (dev < 0)
      {
        GetCmdLineArgument("device", dev);
      }
      if ((dev > deviceCount - 1) || (dev < 0))
      {
        dev = 0;
      }

      error = CubDebug(cudaSetDevice(dev));
      if (error)
      {
        break;
      }

      CubDebugExit(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));

      int ptx_version = 0;
      error           = CubDebug(CUB_NS_QUALIFIER::PtxVersion(ptx_version));
      if (error)
      {
        break;
      }

      error = CubDebug(cudaGetDeviceProperties(&deviceProp, dev));
      if (error)
      {
        break;
      }

      if (deviceProp.major < 1)
      {
        fprintf(stderr, "Device does not support CUDA.\n");
        exit(1);
      }

      int memoryClockRate{};
      error = CubDebug(cudaDeviceGetAttribute(&memoryClockRate, cudaDevAttrMemoryClockRate, dev));
      if (error)
      {
        break;
      }

      int memoryBusWidth{};
      error = CubDebug(cudaDeviceGetAttribute(&memoryBusWidth, cudaDevAttrGlobalMemoryBusWidth, dev));
      if (error)
      {
        break;
      }

      device_giga_bandwidth = float(memoryBusWidth) * memoryClockRate * 2 / 8 / 1000 / 1000;

      if (!CheckCmdLineFlag("quiet"))
      {
        printf(
          "Using device %d: %s (PTX version %d, SM%d, %d SMs, "
          "%lld free / %lld total MB physmem, "
          "%.3f GB/s @ %d kHz mem clock, ECC %s)\n",
          dev,
          deviceProp.name,
          ptx_version,
          deviceProp.major * 100 + deviceProp.minor * 10,
          deviceProp.multiProcessorCount,
          (unsigned long long) device_free_physmem / 1024 / 1024,
          (unsigned long long) device_total_physmem / 1024 / 1024,
          device_giga_bandwidth,
          memoryClockRate,
          (deviceProp.ECCEnabled) ? "on" : "off");
        fflush(stdout);
      }

    } while (0);

    return error;
  }
};
