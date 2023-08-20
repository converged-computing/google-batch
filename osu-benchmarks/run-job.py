from google.cloud import batch_v1
from google.cloud import storage
import os
import jinja2
import argparse

# Script templates

run_script = """#!/bin/bash
export PATH=/opt/intel/mpi/latest/bin:$PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mpi/latest/lib:/opt/intel/mpi/latest/lib/release
find /opt/intel -name mpicc

# This is important - it won't work without sourcing
source /opt/intel/mpi/latest/env/vars.sh

# Job output (per id) goes here
job_output=/mnt/share/{{ job_output }}
echo "Job output will be written to ${job_output}"
mkdir -p ${job_output}

echo "Batch task index is $BATCH_TASK_INDEX"
echo "Batch file is $BATCH_HOSTS_FILE"
cat $BATCH_HOSTS_FILE
output=/mnt/share/{{outdir}}

if [ $BATCH_TASK_INDEX = 0 ]; then
  for name in osu_acc_latency osu_get_acc_latency osu_fop_latency osu_get_latency osu_put_latency osu_allreduce osu_latency osu_bibw osu_bw; do
      latency_exe=$(find . -name ${name})
      latency_exe=$(realpath ${latency_exe})
      echo "${name} is at ${latency_exe}"
      echo "mpirun --hostfile $BATCH_HOSTS_FILE -n {{tasks}} -ppn {{tasks_per_node}} ${latency_exe}"
      mpirun --hostfile $BATCH_HOSTS_FILE -n {{tasks}} -ppn {{tasks_per_node}} ${latency_exe} 2>&1 | tee ${job_output}/${name}.txt
  done
fi
"""

setup_script = """#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
sleep $BATCH_TASK_INDEX

# Note that for this family / image, we are root (do not need sudo)
yum update -y && yum install -y cmake gcc gcc-c++ tuned ethtool git make wget tree

# This ONLY works on the hpc-* image family images
google_mpi_tuning --nosmt
# google_install_mpi --intel_mpi
google_install_intelmpi --impi_2021
source /opt/intel/mpi/latest/env/vars.sh

# This is where they are installed to
# ls /opt/intel/mpi/latest/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/intel/mpi/latest/lib:/opt/intel/mpi/latest/lib/release
export PATH=/opt/intel/mpi/latest/bin:$PATH
echo "Batch task index is ${BATCH_TASK_INDEX}"

echo "Installing OSU benchmarks"
mkdir -p /opt/src
cd /opt/src

function install_osu {
  git clone --depth 1 https://github.com/ULHPC/tutorials ./tutorials && \
  mkdir -p ./osu-benchmark && \
  cd /opt/src/osu-benchmark && \
  ln -s /opt/src/tutorials/parallel/mpi/OSU_MicroBenchmarks ref.d && \
  ln -s ref.d/Makefile . && \
  ln -s ref.d/scripts  . && \
  mkdir src && \
  cd src && \
  export OSU_VERSION=5.8 && \
  wget --no-check-certificate http://mvapich.cse.ohio-state.edu/download/mvapich/osu-micro-benchmarks-${OSU_VERSION}.tgz && \
  tar xf osu-micro-benchmarks-${OSU_VERSION}.tgz && \
  cd $/opt/src/osu-benchmark && \
  # Compile based on openmpi
  mkdir -p build.openmpi && cd build.openmpi && \
  export CFLAGS=\"-I/opt/intel/mpi/latest/include -I/opt/intel/mpi/2021.8.0/include -L/opt/intel/mpi/2021.8.0/lib/release -L/opt/intel/mpi/2021.8.0/lib -I$(pwd)/../src/osu-micro-benchmarks-${OSU_VERSION}/util\"
  ../src/osu-micro-benchmarks-${OSU_VERSION}/configure CC=mpicc CFLAGS=$CFLAGS --prefix=$(pwd) && \
  make && make install
}

install_osu || echo "Cannot install"

echo "        LogLevel ERROR" >> /etc/ssh/ssh_config && \
echo "        StrictHostKeyChecking no" >> /etc/ssh/ssh_config && \
echo "        UserKnownHostsFile=/dev/null" >> /etc/ssh/ssh_config && \
cd /root && \
mkdir -p /run/sshd && \
ssh-keygen -t rsa -N "" -f /root/.ssh/id_rsa && chmod og+rX . && \
cd .ssh && cat id_rsa.pub > authorized_keys && chmod 644 authorized_keys

#mpicc -std=c99 -lmpi -lmpifort -O3 netmark.c -DTRACE -I/opt/intel/mpi/latest/include -I/opt/intel/mpi/2021.8.0/include -L/opt/intel/mpi/2021.8.0/lib/release -L/opt/intel/mpi/2021.8.0/lib -o netmark.x
"""


def get_parser():
    parser = argparse.ArgumentParser(
        description="Hello World OSU Benchmarks",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("project", help="Google cloud project")
    parser.add_argument("--region", help="Google cloud region", default="us-central1")

    # Be careful changing this - the Google MPI won't work on debian
    parser.add_argument(
        "--image-family", help="Google cloud image family", default="hpc-centos-7"
    )
    parser.add_argument(
        "--image-project",
        help="Google cloud image project",
        default="cloud-hpc-image-public",
    )

    parser.add_argument(
        "--job-name",
        dest="job_name",
        help="Name for the job",
        default="osu-benchmarks",
    )
    parser.add_argument("--bucket", help="Bucket to save results to", required=True)
    # mpitune configurations are validated on c2 and c2d instances only.
    parser.add_argument(
        "--machine-type",
        help="Google cloud machine type / VM",
        default="c2-standard-16",
    )
    parser.add_argument(
        "--outdir",
        help="output directory for results (relative to bucket)",
        default="osu-benchmarks",
    )
    parser.add_argument("--retry-count", help="retry count", type=int, default=1)
    parser.add_argument(
        "--tasks-per-node",
        help="number of Batch tasks to assign per node",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max-run-duration",
        help="maximum run duration, string (e.g., 3600s)",
        default="3600s",
    )  # 1GB
    parser.add_argument(
        "--cpu-milli", help="milliseconds per cpu-second", type=int, default=1000
    )
    parser.add_argument("--memory", help="memory in MiB", type=int, default=1000)  # 1GB
    return parser


# pip install --upgrade google-cloud-storage.
def upload_to_bucket(blob_name, path_to_file, bucket_name):
    """
    Upload script file to a bucket
    """
    storage_client = storage.Client()

    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(path_to_file)
    return blob.public_url


def main():
    """
    Run an osu benchmark experiment using Google batch.
    """
    parser = get_parser()

    # If an error occurs while parsing the arguments, the interpreter will exit with value 2
    args, _ = parser.parse_known_args()

    # Let's write one big stupid function like in the example!
    job = batch_job(
        args.project,
        args.region,
        args.job_name,
        bucket_name=args.bucket,
        machine_type=args.machine_type,
        image_family=args.image_family,
        image_project=args.image_project,
        outdir=args.outdir,
        cpu_milli=args.cpu_milli,
        memory=args.memory,
        retry_count=args.retry_count,
        max_run_duration=args.max_run_duration,
        tasks_per_node=args.tasks_per_node,
    )

    print("Submit Job!")
    print(job)


def template_and_write(local_path, template, bucket_name, bucket_path=None):
    """
    Write script to a local path, and upload to bucket
    """
    bucket_path = bucket_path or local_path
    with open(local_path, "w") as fd:
        fd.write(template)
    upload_to_bucket(bucket_path, local_path, bucket_name)


def batch_job(
    project_id: str,
    region: str,
    job_name: str,
    bucket_name: str,
    outdir: str,
    retry_count: int,
    tasks_per_node: int,
    machine_type: str,
    image_family: str,
    memory: int,
    cpu_milli: int,
    image_project: str,
    max_run_duration: str,
) -> batch_v1.Job:
    """
    Create a Netmark Job, building netmark from mounted code in a Google Bucket.
    """
    client = batch_v1.BatchServiceClient()

    # Output directory specific to job name
    job_output = os.path.join(outdir, job_name)

    # Prepare the script templates
    run_template = jinja2.Template(run_script)
    runscript = run_template.render(
        {
            "tasks": 2,
            "tasks_per_node": tasks_per_node,
            "outdir": outdir,
        }
    )

    setup_template = jinja2.Template(setup_script)
    script = setup_template.render({"outdir": outdir, "job_output": job_output})

    # Storage subdir name
    subdir = "osu-benchmarks"

    # Write over same file here, yes bad practice and lazy
    template_and_write("setup.sh", script, bucket_name, f"{subdir}/setup.sh")
    template_and_write("run.sh", runscript, bucket_name, f"{subdir}/run.sh")

    # Define what will be done as part of the job.
    task = batch_v1.TaskSpec()
    setup = batch_v1.Runnable()
    setup.script = batch_v1.Runnable.Script()
    setup.script.text = f"bash /mnt/share/{subdir}/setup.sh"

    # This will ensure all nodes finish first
    barrier = batch_v1.Runnable()
    barrier.barrier = batch_v1.Runnable.Barrier()
    barrier.barrier.name = "wait-for-setup"

    runnable = batch_v1.Runnable()
    runnable.script = batch_v1.Runnable.Script()
    runnable.script.text = f"bash /mnt/share/{subdir}/run.sh"
    task.runnables = [setup, barrier, runnable]

    gcs_bucket = batch_v1.GCS()
    gcs_bucket.remote_path = bucket_name
    gcs_volume = batch_v1.Volume()
    gcs_volume.gcs = gcs_bucket
    gcs_volume.mount_path = "/mnt/share"
    task.volumes = [gcs_volume]

    # We can specify what resources are requested by each task.
    resources = batch_v1.ComputeResource()

    # in milliseconds per cpu-second. The requirement % of a single CPUs.
    resources.cpu_milli = cpu_milli
    resources.memory_mib = memory
    task.compute_resource = resources

    task.max_retry_count = retry_count
    task.max_run_duration = max_run_duration

    # Tasks are grouped inside a job using TaskGroups.
    # Currently, it's possible to have only one task group.
    group = batch_v1.TaskGroup()
    group.task_count_per_node = tasks_per_node
    group.task_count = 2
    group.task_spec = task
    group.require_hosts_file = True
    group.permissive_ssh = True

    # Disks is how we specify the image we want (we need centos to get MPI working)
    # This could also be batch-centos, batch-debian, batch-cos but MPI won't work on all of those
    boot_disk = batch_v1.AllocationPolicy.Disk()
    boot_disk.image = f"projects/{image_project}/global/images/family/{image_family}"

    # Policies are used to define on what kind of virtual machines the tasks will run on.
    # https://github.com/googleapis/python-batch/blob/main/google/cloud/batch_v1/types/job.py#L383
    allocation_policy = batch_v1.AllocationPolicy()
    policy = batch_v1.AllocationPolicy.InstancePolicy()
    policy.machine_type = machine_type
    policy.boot_disk = boot_disk

    instances = batch_v1.AllocationPolicy.InstancePolicyOrTemplate()
    instances.policy = policy
    allocation_policy.instances = [instances]

    job = batch_v1.Job()
    job.task_groups = [group]
    job.allocation_policy = allocation_policy
    job.labels = {"env": "testing", "type": "script", "mount": "bucket"}

    # We use Cloud Logging as it's an out of the box available option
    job.logs_policy = batch_v1.LogsPolicy()
    job.logs_policy.destination = batch_v1.LogsPolicy.Destination.CLOUD_LOGGING

    create_request = batch_v1.CreateJobRequest()
    create_request.job = job
    create_request.job_id = job_name

    # The job's parent is the region in which the job will run
    create_request.parent = f"projects/{project_id}/locations/{region}"
    return client.create_job(create_request)


if __name__ == "__main__":
    main()
