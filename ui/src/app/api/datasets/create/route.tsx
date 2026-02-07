import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';
import { getDatasetsRoot } from '@/server/settings';

export async function POST(request: Request) {
  try {
    const body = await request.json();
    let { name } = body;
    // clean name by making lower case,  removing special characters, and replacing spaces with underscores
    // Only remove characters disallowed by the operating system (Windows/Linux)
    // Removed .toLowerCase() to preserve the original casing of the input
    name = name.trim().replace(/[<>:"/\\|?*]+/g, '_');

    let datasetsPath = await getDatasetsRoot();
    let datasetPath = path.join(datasetsPath, name);

    // if folder doesnt exist, create it
    if (!fs.existsSync(datasetPath)) {
      fs.mkdirSync(datasetPath);
    }

    return NextResponse.json({ success: true, name: name });
  } catch (error) {
    return NextResponse.json({ error: 'Failed to create dataset' }, { status: 500 });
  }
}
