"""Linux OS boundary: private namespaces followed by a syscall allowlist."""
from __future__ import annotations

import ctypes
import errno
import os
from pathlib import Path
import resource
import shutil

from protocol import ContractError, OperationalError

ROOT = Path(__file__).resolve().parent
# No file-open, path metadata, sockets, process creation, exec, ptrace,
# process_vm, mount, io_uring, descriptor duplication or device ioctl.
ALLOW_SYSCALLS = (
    "read", "write", "readv", "writev", "close", "fstat", "lseek",
    "futex", "futex_time64", "mmap", "mmap2", "mprotect", "munmap", "brk",
    "madvise", "mremap", "rt_sigaction", "rt_sigprocmask", "rt_sigreturn",
    "sigaltstack", "getpid", "gettid", "getuid", "getgid", "geteuid", "getegid",
    "clock_gettime", "clock_gettime64", "gettimeofday", "times", "getrusage",
    "sched_yield", "exit", "exit_group", "setitimer", "alarm", "nanosleep",
    "clock_nanosleep", "clock_nanosleep_time64", "getrandom", "restart_syscall",
)


def namespace_identity():
    return {name: os.readlink("/proc/self/ns/" + name) for name in ("mnt", "pid", "net", "user")}


def linux_command(python, bootstrap):
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise OperationalError("sandbox_unavailable")
    executable = Path(python).absolute()
    if not executable.is_file():
        raise OperationalError("sandbox_python_missing")
    # The selected host/sandbox interpreter is independent of the simulator's
    # Python 3.9. Bind its prefix and resolved executable prefix for venvs.
    prefixes = {executable.parent.parent, executable.resolve().parent.parent}
    command = [bwrap, "--unshare-all", "--die-with-parent", "--new-session",
               "--cap-drop", "ALL", "--proc", "/proc", "--dev", "/dev",
               "--tmpfs", "/tmp", "--chdir", "/tmp"]
    for directory in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(directory).exists():
            command += ["--ro-bind", directory, directory]
    if Path("/etc/ld.so.cache").is_file():
        command += ["--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache"]
    for prefix in sorted(prefixes):
        if str(prefix) not in {"/usr", "/"}:
            command += ["--ro-bind", str(prefix), str(prefix)]
    # Trusted imports only. Source/packet remain inert until seccomp succeeds.
    command += ["--ro-bind", str(ROOT), str(ROOT),
                "--ro-bind", str(ROOT.parent / "src"), str(ROOT.parent / "src")]
    for name, value in namespace_identity().items():
        command += ["--setenv", "EMBODIED_PARENT_NS_" + name.upper(), value]
    command += ["--setenv", "TMPDIR", "/tmp", str(executable), "-I", "-B", "-c", bootstrap]
    return command


def seal_linux():
    identities = namespace_identity()
    if any(not os.environ.get("EMBODIED_PARENT_NS_" + name.upper())
           or value == os.environ["EMBODIED_PARENT_NS_" + name.upper()]
           for name, value in identities.items()):
        raise ContractError("sandbox_namespace_unverified")
    library = None
    for path in ("libseccomp.so.2", "/lib/x86_64-linux-gnu/libseccomp.so.2",
                 "/lib/aarch64-linux-gnu/libseccomp.so.2"):
        try:
            library = ctypes.CDLL(path, use_errno=True)
            break
        except OSError:
            pass
    if library is None:
        raise ContractError("sandbox_seccomp_unavailable")
    library.seccomp_init.argtypes = [ctypes.c_uint32]
    library.seccomp_init.restype = ctypes.c_void_p
    library.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    library.seccomp_syscall_resolve_name.restype = ctypes.c_int
    library.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32,
                                       ctypes.c_int, ctypes.c_uint]
    library.seccomp_rule_add.restype = ctypes.c_int
    library.seccomp_load.argtypes = [ctypes.c_void_p]
    library.seccomp_load.restype = ctypes.c_int
    library.seccomp_release.argtypes = [ctypes.c_void_p]
    # Kernel clears privilege escalation on filter installation; require it
    # explicitly before trusting the filter.
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        raise ContractError("sandbox_no_new_privileges_failed")
    context = library.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise ContractError("sandbox_seccomp_init_failed")
    try:
        for name in ALLOW_SYSCALLS:
            number = library.seccomp_syscall_resolve_name(name.encode())
            if number >= 0 and library.seccomp_rule_add(context, 0x7FFF0000, number, 0) != 0:
                raise ContractError("sandbox_seccomp_rule_failed")
        hard = resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        os.closerange(3, min(int(hard), 1_048_576))
        if library.seccomp_load(context) != 0:
            raise ContractError("sandbox_seccomp_load_failed")
    finally:
        library.seccomp_release(context)
    return {"backend": "linux-bwrap-seccomp/1", "namespaces_verified": True,
            "syscall_allowlist": True, "inherited_fds_closed": True}
