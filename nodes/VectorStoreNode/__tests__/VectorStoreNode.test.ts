import { QueryResult, UpsertDocument, VectorStoreNode, embedText, upsertDocument, queryDocuments, deleteDocument } from '../VectorStoreNode.node';
import type { IExecuteFunctions } from 'n8n-workflow';

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------

// Mock OpenAI
jest.mock('openai', () => {
  return {
    __esModule: true,
    default: jest.fn().mockImplementation(() => ({
      embeddings: {
        create: jest.fn().mockResolvedValue({
          data: [{ embedding: [0.1, 0.2, 0.3, 0.4, 0.5] }],
        }),
      },
    })),
  };
});

// Mock chromadb
const mockCollection = {
  upsert: jest.fn().mockResolvedValue(undefined),
  query: jest.fn(),
  delete: jest.fn().mockResolvedValue(undefined),
};

jest.mock('chromadb', () => ({
  ChromaClient: jest.fn().mockImplementation(() => ({
    getOrCreateCollection: jest.fn().mockResolvedValue(mockCollection),
    listCollections: jest.fn().mockResolvedValue(['col1', 'col2']),
  })),
  IncludeEnum: {
    Documents: 'documents',
    Metadatas: 'metadatas',
    Distances: 'distances',
  },
}));

// ---------------------------------------------------------------------------
// embedText
// ---------------------------------------------------------------------------

describe('embedText', () => {
  beforeEach(() => jest.clearAllMocks());

  it('returns a numeric array', async () => {
    const result = await embedText('test text', 'sk-test-key');
    expect(Array.isArray(result)).toBe(true);
    expect(result.every((v) => typeof v === 'number')).toBe(true);
  });

  it('returns the embedding from the mock', async () => {
    const result = await embedText('hello', 'sk-test');
    expect(result).toEqual([0.1, 0.2, 0.3, 0.4, 0.5]);
  });

  it('uses the default model when not specified', async () => {
    const OpenAIMock = jest.requireMock('openai').default;
    await embedText('text', 'key');
    const instance = OpenAIMock.mock.results[0].value;
    expect(instance.embeddings.create).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'text-embedding-3-small' }),
    );
  });

  it('forwards a custom model to the OpenAI client', async () => {
    const OpenAIMock = jest.requireMock('openai').default;
    await embedText('text', 'key', 'text-embedding-3-large');
    const instance = OpenAIMock.mock.results[0].value;
    expect(instance.embeddings.create).toHaveBeenCalledWith(
      expect.objectContaining({ model: 'text-embedding-3-large' }),
    );
  });
});

// ---------------------------------------------------------------------------
// upsertDocument
// ---------------------------------------------------------------------------

describe('upsertDocument', () => {
  beforeEach(() => jest.clearAllMocks());

  const makeDoc = (overrides: Partial<UpsertDocument> = {}): UpsertDocument => ({
    id: 'doc-001',
    text: 'Quarterly revenue report',
    metadata: { source: 'erp' },
    ...overrides,
  });

  it('returns the document id', async () => {
    const id = await upsertDocument(mockCollection as never, makeDoc(), 'sk-key');
    expect(id).toBe('doc-001');
  });

  it('calls collection.upsert with the embedded vector', async () => {
    await upsertDocument(mockCollection as never, makeDoc(), 'sk-key');
    expect(mockCollection.upsert).toHaveBeenCalledWith(
      expect.objectContaining({
        ids: ['doc-001'],
        embeddings: [[0.1, 0.2, 0.3, 0.4, 0.5]],
        documents: ['Quarterly revenue report'],
        metadatas: [{ source: 'erp' }],
      }),
    );
  });

  it('uses empty metadata when none provided', async () => {
    await upsertDocument(mockCollection as never, { id: 'x', text: 'y' }, 'sk-key');
    expect(mockCollection.upsert).toHaveBeenCalledWith(
      expect.objectContaining({ metadatas: [{}] }),
    );
  });
});

// ---------------------------------------------------------------------------
// queryDocuments
// ---------------------------------------------------------------------------

describe('queryDocuments', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockCollection.query.mockResolvedValue({
      ids: [['id-1', 'id-2']],
      documents: [['First doc text', 'Second doc text']],
      metadatas: [[{ category: 'finance' }, { category: 'legal' }]],
      distances: [[0.12, 0.34]],
    });
  });

  it('returns an array of QueryResult objects', async () => {
    const results = await queryDocuments(mockCollection as never, 'revenue', 2, 'sk-key');
    expect(results).toHaveLength(2);
  });

  it('maps ids, documents, metadatas and distances correctly', async () => {
    const results = await queryDocuments(mockCollection as never, 'revenue', 2, 'sk-key');
    expect(results[0]).toEqual<QueryResult>({
      id: 'id-1',
      document: 'First doc text',
      metadata: { category: 'finance' },
      distance: 0.12,
    });
    expect(results[1]).toEqual<QueryResult>({
      id: 'id-2',
      document: 'Second doc text',
      metadata: { category: 'legal' },
      distance: 0.34,
    });
  });

  it('passes nResults equal to topK', async () => {
    await queryDocuments(mockCollection as never, 'q', 7, 'sk-key');
    expect(mockCollection.query).toHaveBeenCalledWith(
      expect.objectContaining({ nResults: 7 }),
    );
  });

  it('handles empty result sets gracefully', async () => {
    mockCollection.query.mockResolvedValueOnce({
      ids: [[]],
      documents: [[]],
      metadatas: [[]],
      distances: [[]],
    });
    const results = await queryDocuments(mockCollection as never, 'q', 5, 'sk-key');
    expect(results).toHaveLength(0);
  });

  it('handles missing distances field gracefully', async () => {
    mockCollection.query.mockResolvedValueOnce({
      ids: [['id-x']],
      documents: [['doc']],
      metadatas: [[{}]],
    });
    const results = await queryDocuments(mockCollection as never, 'q', 1, 'sk-key');
    expect(results[0].distance).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// deleteDocument
// ---------------------------------------------------------------------------

describe('deleteDocument', () => {
  beforeEach(() => jest.clearAllMocks());

  it('calls collection.delete with the correct id', async () => {
    await deleteDocument(mockCollection as never, 'doc-xyz');
    expect(mockCollection.delete).toHaveBeenCalledWith({ ids: ['doc-xyz'] });
  });

  it('resolves without throwing on success', async () => {
    await expect(deleteDocument(mockCollection as never, 'doc-abc')).resolves.toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// VectorStoreNode.execute() — integration tests
// ---------------------------------------------------------------------------

function makeVSExecuteFunctions(params: Record<string, unknown>, continueOnFail = false): IExecuteFunctions {
  return {
    getInputData: () => [{ json: {} }],
    getNodeParameter: (name: string) => {
      if (name in params) return params[name];
      return '';
    },
    continueOnFail: () => continueOnFail,
    getNode: () => ({ name: 'VectorStoreNode', type: 'vectorStoreNode' } as never),
  } as unknown as IExecuteFunctions;
}

describe('VectorStoreNode.execute()', () => {
  const node = new VectorStoreNode();

  beforeEach(() => {
    jest.clearAllMocks();
    mockCollection.query.mockResolvedValue({
      ids: [['r1']],
      documents: [['Doc text']],
      metadatas: [[{ src: 'test' }]],
      distances: [[0.1]],
    });
  });

  it('upsert: calls collection.upsert and returns id', async () => {
    const ctx = makeVSExecuteFunctions({
      operation: 'upsert',
      chromaUrl: 'http://localhost:8000',
      collectionName: 'test-col',
      documentId: 'doc-1',
      text: 'Hello world',
      metadata: '{}',
      embeddingModel: 'text-embedding-3-small',
      openAiApiKey: 'sk-test',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.success).toBe(true);
    expect(result.json.id).toBe('doc-1');
  });

  it('query: returns results array', async () => {
    const ctx = makeVSExecuteFunctions({
      operation: 'query',
      chromaUrl: 'http://localhost:8000',
      collectionName: 'test-col',
      queryText: 'What is revenue?',
      topK: 1,
      embeddingModel: 'text-embedding-3-small',
      openAiApiKey: 'sk-test',
    });
    const [[result]] = await node.execute.call(ctx);
    expect((result.json.results as QueryResult[]).length).toBeGreaterThanOrEqual(0);
    expect(result.json.count).toBeDefined();
  });

  it('delete: calls collection.delete', async () => {
    const ctx = makeVSExecuteFunctions({
      operation: 'delete',
      chromaUrl: 'http://localhost:8000',
      collectionName: 'test-col',
      documentId: 'doc-del',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.success).toBe(true);
    expect(result.json.deletedId).toBe('doc-del');
  });

  it('listCollections: returns collection names', async () => {
    const ctx = makeVSExecuteFunctions({
      operation: 'listCollections',
      chromaUrl: 'http://localhost:8000',
    });
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.collections).toBeDefined();
  });

  it('unknown operation: throws without continueOnFail', async () => {
    const ctx = makeVSExecuteFunctions({
      operation: 'unknown',
      chromaUrl: 'http://localhost:8000',
    });
    await expect(node.execute.call(ctx)).rejects.toThrow();
  });

  it('error with continueOnFail: returns error object', async () => {
    mockCollection.upsert.mockRejectedValueOnce(new Error('ChromaDB unavailable'));
    const ctx = makeVSExecuteFunctions({
      operation: 'upsert',
      chromaUrl: 'http://localhost:8000',
      collectionName: 'test-col',
      documentId: 'doc-err',
      text: 'text',
      metadata: '{}',
      embeddingModel: 'text-embedding-3-small',
      openAiApiKey: 'sk-test',
    }, true);
    const [[result]] = await node.execute.call(ctx);
    expect(result.json.error).toBeDefined();
  });
});
