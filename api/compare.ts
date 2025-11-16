import type { VercelRequest, VercelResponse } from '@vercel/node';
import * as faceapi from '@vladmandic/face-api';
import { Canvas, Image } from '@napi-rs/canvas';
import { join } from 'path';
// @ts-ignore
faceapi.env.monkeyPatch({ Canvas, Image });
let modelsLoaded = false;
async function loadModels() {
  if (modelsLoaded) return;
  const modelPath = join(process.cwd(), 'models');
  await Promise.all([
    faceapi.nets.ssdMobilenetv1.loadFromDisk(modelPath),
    faceapi.nets.faceLandmark68Net.loadFromDisk(modelPath),
    faceapi.nets.faceRecognitionNet.loadFromDisk(modelPath)
  ]);
  modelsLoaded = true;
}
async function getDescriptor(base64: string) {
  const img = new Image();
  img.src = base64;
  const detection = await faceapi.detectSingleFace(img as any).withFaceLandmarks().withFaceDescriptor();
  return detection?.descriptor;
}
export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') return res.status(405).json({ error: 'Method not allowed' });
  try {
    console.log('Loading models...');
    await loadModels();
    console.log('Models loaded');
    const { image1, image2 } = req.body;
    if (!image1 || !image2) return res.status(400).json({ error: 'Two images required' });
    console.log('Processing images...');
    const [desc1, desc2] = await Promise.all([getDescriptor(image1), getDescriptor(image2)]);
    if (!desc1 || !desc2) return res.status(400).json({ error: 'No face detected in one or both images' });
    const distance = faceapi.euclideanDistance(desc1, desc2);
    const similarity = Math.max(0, Math.min(100, (1 - distance) * 100));
    console.log('Similarity:', similarity);
    res.json({ similarity: Math.round(similarity) });
  } catch (error) {
    console.error('Error:', error);
    res.status(500).json({ error: error instanceof Error ? error.message : 'Processing failed' });
  }
}
