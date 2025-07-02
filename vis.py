import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_outputs(input_dir='normalized', output_dir='normalizedplots'):
    """
    Walk through the input directory structure, parse JSON files, extract object positions,
    and save 3D scatter plots organized by model and category. Axis limits are auto-adjusted
    around the data with padding, and output images are generated at high resolution.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Colormap for distinct object colors
    cmap = plt.get_cmap('tab20')

    # Traverse models and categories
    for model_dir in input_path.iterdir():
        if not model_dir.is_dir():
            continue
        model = model_dir.name
        for category_dir in model_dir.iterdir():
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            # Prepare output directory
            save_dir = output_path / model / category
            save_dir.mkdir(parents=True, exist_ok=True)

            # Process each JSON file
            for json_file in category_dir.glob('*.json'):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception as e:
                    print(f"Failed to read {json_file}: {e}")
                    continue

                raw = data.get('raw_output', '')
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    print(f"Failed to parse raw_output in {json_file}: {e}")
                    continue

                objects = parsed.get('objects', {})
                if not objects:
                    continue

                # Gather positions
                labels, xs, ys, zs = [], [], [], []
                for name, info in objects.items():
                    pos = info.get('position', {})
                    x, y, z = pos.get('x'), pos.get('y'), pos.get('z')
                    if None in (x, y, z):
                        continue
                    labels.append(name)
                    xs.append(x)
                    ys.append(y)
                    zs.append(z)

                if not xs:
                    continue

                # Compute axis bounds with padding
                min_x, max_x = min(xs), max(xs)
                min_y, max_y = min(ys), max(ys)
                min_z, max_z = min(zs), max(zs)
                range_x = max_x - min_x
                range_y = max_y - min_y
                range_z = max_z - min_z
                max_range = max(range_x, range_y, range_z)
                mid_x = (max_x + min_x) / 2
                mid_y = (max_y + min_y) / 2
                mid_z = (max_z + min_z) / 2
                pad_factor = 0.1  # 10% padding
                half_span = max_range / 2 * (1 + pad_factor)

                # Create high-resolution figure
                fig = plt.figure(figsize=(10, 8), dpi=200)
                ax = fig.add_subplot(111, projection='3d')

                # Plot each object with distinct color
                for idx, (label, x, y, z) in enumerate(zip(labels, xs, ys, zs)):
                    color = cmap(idx % cmap.N)
                    ax.scatter(x, y, z, color=color, label=label)
                    ax.text(x, y, z, f" {label}")

                # Apply auto-scaled limits
                ax.set_xlim(mid_x - half_span, mid_x + half_span)
                ax.set_ylim(mid_y - half_span, mid_y + half_span)
                ax.set_zlim(mid_z - half_span, mid_z + half_span)

                # Labels and legend
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
                ax.legend(loc='upper right')

                # Title and save
                title = data.get('scenario_id', json_file.stem)
                ax.set_title(title)
                out_file = save_dir / f"{title}.png"
                try:
                    fig.savefig(out_file)
                    print(f"Saved visualization to {out_file}")
                except Exception as e:
                    print(f"Failed to save figure for {json_file}: {e}")
                finally:
                    plt.close(fig)


if __name__ == '__main__':
    visualize_outputs()
