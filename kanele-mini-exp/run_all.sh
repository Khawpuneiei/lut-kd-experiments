#!/usr/bin/env bash
# Runs every step of README.md on a 4-core CPU: teachers, then Fashion-MNIST and CIFAR-10 students
# (ce vs kd, 3 seeds), then the CIFAR-10 pruning sweep. Two jobs at a time, 2 threads each.
# Teachers run from a scratch directory so the pilot's results/teacher_*.json are not overwritten.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PILOT="$HERE/../dwn-kd-pilot"
CACHE="${CACHE_DIR:-/home/claude/work/cache}"
mkdir -p "$HERE/results/logs"
cd "$HERE"

pair() {  # run two commands in parallel, each logging to its own file
  local a="$1" la="$2" b="$3" lb="$4"
  echo "START $(date +%T) $la | $lb"
  bash -c "$a" > "results/logs/$la.log" 2>&1 &
  bash -c "$b" > "results/logs/$lb.log" 2>&1 &
  wait
  echo "DONE  $(date +%T) $la: $(tail -1 results/logs/$la.log | cut -c1-160)"
  echo "DONE  $(date +%T) $lb: $(tail -1 results/logs/$lb.log | cut -c1-160)"
}

TDIR="$(mktemp -d)"
T="cd $TDIR && python3 -W ignore $PILOT/teacher.py --threads 2"
pair "$T --dataset fmnist --epochs 12" teacher_fmnist "$T --dataset cifar10 --epochs 15" teacher_cifar10
cp "$TDIR"/results/teacher_*.json results/

S="python3 -W ignore train_lutkan.py --threads 2 --device cpu"
for s in 0 1 2; do
  pair "$S --dataset fmnist --mode ce --epochs 10 --seed $s --out results/fmnist.jsonl" fmnist_ce_s$s \
       "$S --dataset fmnist --mode kd --epochs 10 --T 2 --alpha 0.9 --seed $s --out results/fmnist.jsonl" fmnist_kd_s$s
done
C="$S --dataset cifar10 --hidden 64 --in_bits 2 --act_bits 4 --epochs 10"
for s in 0 1 2; do
  pair "$C --mode ce --seed $s --out results/cifar10.jsonl" cifar10_ce_s$s \
       "$C --mode kd --T 2 --alpha 0.9 --seed $s --out results/cifar10.jsonl" cifar10_kd_s$s
done
P="$C --mode kd --T 2 --alpha 0.9 --seed 0 --t0 3 --tf 8 --tag prune --out results/cifar10_prune.jsonl"
pair "$P --prune 0.1" prune_0.1 "$P --prune 0.2" prune_0.2
pair "$P --prune 0.3" prune_0.3 "$P --prune 0.5" prune_0.5
echo "ALL DONE $(date +%T)"
