"""
Profiling script for demo_phase_03.py to identify bottlenecks.

Usage:
    python profile_demo_03.py --date 2018-10-18
"""

import cProfile
import pstats
import io
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def profile_demo():
    """Run demo_phase_03.py with profiling enabled."""
    
    # Import demo module
    sys.path.insert(0, str(Path(__file__).parent / "workspace"))
    import demo_phase_03
    
    # Prepare arguments
    sys.argv = [
        'demo_phase_03.py',
        '--date', '2018-10-18',
        # '--visualize',  # Comment out to skip visualization for faster profiling
        # '--viz-dir', 'visualizations_profile'
    ]
    
    # Create profiler
    profiler = cProfile.Profile()
    
    print("=" * 80)
    print("Starting profiling of demo_phase_03.py...")
    print("=" * 80)
    
    # Run with profiling
    profiler.enable()
    try:
        demo_phase_03.main()
    except SystemExit:
        pass
    profiler.disable()
    
    print("\n" + "=" * 80)
    print("Profiling Results - Top 30 Time-Consuming Functions")
    print("=" * 80)
    
    # Print results sorted by cumulative time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(30)
    print(s.getvalue())
    
    print("\n" + "=" * 80)
    print("Profiling Results - Top 30 by Total Time (excluding subcalls)")
    print("=" * 80)
    
    # Print results sorted by total time
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('tottime')
    ps.print_stats(30)
    print(s.getvalue())
    
    # Print specific bottleneck analysis
    print("\n" + "=" * 80)
    print("Bottleneck Analysis")
    print("=" * 80)
    
    # Get stats
    ps = pstats.Stats(profiler)
    
    # Find functions with most cumulative time
    stats = ps.stats
    sorted_stats = sorted(stats.items(), key=lambda x: x[1][3], reverse=True)[:10]
    
    print("\nTop 10 functions by cumulative time:")
    print(f"{'Function':<60} {'Cumtime (s)':<12} {'Calls':<10}")
    print("-" * 80)
    for (filename, line, func_name), (cc, nc, tt, ct, callers) in sorted_stats:
        func_display = f"{Path(filename).name}:{line}({func_name})"[:60]
        print(f"{func_display:<60} {ct:<12.3f} {nc:<10}")
    
    # Save detailed profile to file
    profile_file = Path(__file__).parent / "profile_demo_03.prof"
    profiler.dump_stats(str(profile_file))
    print(f"\n✓ Detailed profile saved to: {profile_file}")
    print(f"  View with: python -m pstats {profile_file}")
    
    # Generate call graph hint
    print("\nTo generate a call graph visualization:")
    print(f"  pip install gprof2dot")
    print(f"  gprof2dot -f pstats {profile_file} | dot -Tpng -o profile_callgraph.png")


if __name__ == "__main__":
    profile_demo()
