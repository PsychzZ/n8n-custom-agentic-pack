import {
  IExecuteFunctions,
  INodeExecutionData,
  INodeType,
  INodeTypeDescription,
  NodeOperationError,
} from 'n8n-workflow';
import { ChromaClient, Collection, Metadata } from 'chromadb';
import OpenAI from 'openai';

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/**
 * @description Represents a single document returned by a ChromaDB similarity
 * search. Mirrors the ChromaDB query-result structure but with explicit typing
 * for downstream n8n node consumption.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation for
 * Knowledge-Intensive NLP Tasks" - Section 2.1 defines the retriever component
 * as returning a set of documents D = {d_1, ..., d_k} with associated metadata.
 * This interface maps directly to one such document d_i.
 *
 * @thesis_note DE: QueryResult bildet das Rückgabeformat des Retrievers ab.
 * Die Felder id, document, metadata und distance entsprechen den in Lewis et al.
 * (2020) beschriebenen Retriever-Ausgaben und ermöglichen eine direkte Integration
 * in den RAG-Prompt des ReActLoopNode.
 */
export interface QueryResult {
  /** ChromaDB document identifier. */
  id: string;
  /** Original text content of the stored document. */
  document: string;
  /** Arbitrary key-value metadata associated with the document. */
  metadata: Record<string, unknown>;
  /** Cosine (or L2/IP) distance to the query vector; lower = more similar. */
  distance: number;
}

/**
 * @description Configuration for a single upsert (embed + store) operation.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * The document store is populated offline (indexing phase) and queried online
 * (retrieval phase). UpsertDocument represents the indexing-phase payload.
 *
 * @thesis_note DE: UpsertDocument kapselt alle Informationen, die benötigt
 * werden, um ein Dokument in den Vektorspeicher zu integrieren. Die optionalen
 * Metadaten erlauben spätere filterbasierte Suchen.
 */
export interface UpsertDocument {
  /** Unique document identifier (UUID recommended). */
  id: string;
  /** Text to embed and store. */
  text: string;
  /** Optional metadata for filtered retrieval. */
  metadata?: Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// Embedding helper
// ---------------------------------------------------------------------------

/**
 * @description Generates a dense vector embedding for the given text using the
 * OpenAI Embeddings API with the `text-embedding-3-small` model.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * Dense retrieval via bi-encoder embeddings is the standard approach for
 * scalable semantic search in RAG systems.
 *
 * @param text     - The text to embed.
 * @param apiKey   - OpenAI API key.
 * @param model    - Embedding model identifier (default: text-embedding-3-small).
 * @returns A float array representing the embedding vector.
 *
 * @thesis_note DE: Die Embedding-Funktion kapselt den OpenAI API-Aufruf und
 * ist bewusst als separate Funktion extrahiert, um die Austauschbarkeit des
 * Embedding-Modells zu gewährleisten (z.B. lokales Modell via Ollama).
 *
 * @example
 * const vec = await embedText('quarterly revenue report', process.env.OPENAI_API_KEY!);
 */
export async function embedText(
  text: string,
  apiKey: string,
  model = 'text-embedding-3-small',
): Promise<number[]> {
  const client = new OpenAI({ apiKey });
  const response = await client.embeddings.create({ model, input: text });
  return response.data[0].embedding;
}

// ---------------------------------------------------------------------------
// ChromaDB operation helpers (independently testable)
// ---------------------------------------------------------------------------

/**
 * @description Embeds a document and upserts it into the specified ChromaDB
 * collection. If a document with the same id already exists it is overwritten.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * Upsert semantics (create-or-update) keep the document store current without
 * manual delete-before-insert sequences, matching the "document indexer" role
 * described in the RAG architecture.
 *
 * @param collection - An active ChromaDB Collection instance.
 * @param doc        - The document to embed and store.
 * @param apiKey     - OpenAI API key for embedding generation.
 * @param model      - Embedding model to use.
 * @returns The id of the upserted document.
 *
 * @thesis_note DE: upsertDocument implementiert die Indexierungsphase des RAG-
 * Prozesses. Die Upsert-Semantik vermeidet Duplikate und ermöglicht inkrementelle
 * Aktualisierungen des Wissensspeichers, was für produktive ERP-Integrationen
 * (SC-01) besonders relevant ist.
 *
 * @example
 * const id = await upsertDocument(collection, {
 *   id: 'doc-001',
 *   text: 'Q2 revenue was €2.4M.',
 *   metadata: { source: 'erp', quarter: 'Q2' },
 * }, apiKey);
 */
export async function upsertDocument(
  collection: Collection,
  doc: UpsertDocument,
  apiKey: string,
  model = 'text-embedding-3-small',
): Promise<string> {
  const embedding = await embedText(doc.text, apiKey, model);
  await collection.upsert({
    ids: [doc.id],
    embeddings: [embedding],
    documents: [doc.text],
    metadatas: [(doc.metadata ?? {}) as Metadata],
  });
  return doc.id;
}

/**
 * @description Performs a semantic similarity search against the ChromaDB
 * collection and returns the top-k most similar documents.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * Top-k retrieval (Section 2.1) is the core retrieval mechanism of RAG.
 * The retrieved documents D are concatenated into the LLM context window.
 *
 * @param collection - An active ChromaDB Collection instance.
 * @param queryText  - The natural-language query to embed and search.
 * @param topK       - Number of results to return (default: 5).
 * @param apiKey     - OpenAI API key for query embedding.
 * @param model      - Embedding model for the query.
 * @returns An array of up to topK QueryResult objects sorted by ascending distance.
 *
 * @thesis_note DE: queryDocuments implementiert die Retrieval-Phase des RAG-
 * Systems. Das topK-Argument entspricht dem k-Parameter aus Lewis et al. (2020).
 * Der SC-02-Benchmark misst die Abrufgenauigkeit (Precision@k) dieser Funktion.
 *
 * @example
 * const results = await queryDocuments(collection, 'Q2 revenue', 3, apiKey);
 * // results[0].document => most relevant text passage
 */
export async function queryDocuments(
  collection: Collection,
  queryText: string,
  topK: number,
  apiKey: string,
  model = 'text-embedding-3-small',
): Promise<QueryResult[]> {
  const queryEmbedding = await embedText(queryText, apiKey, model);

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: topK,
    include: ['documents', 'metadatas', 'distances'] as any,
  });

  const ids = results.ids[0] ?? [];
  const documents = results.documents[0] ?? [];
  const metadatas = results.metadatas[0] ?? [];
  const distances = results.distances?.[0] ?? [];

  return ids.map((id, idx) => ({
    id,
    document: documents[idx] ?? '',
    metadata: (metadatas[idx] ?? {}) as Record<string, unknown>,
    distance: distances[idx] ?? 0,
  }));
}

/**
 * @description Removes a document from the ChromaDB collection by its id.
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
 * Maintaining a clean, up-to-date knowledge base requires the ability to remove
 * stale or incorrect documents (document lifecycle management).
 *
 * @param collection - An active ChromaDB Collection instance.
 * @param id         - The document id to remove.
 * @returns void
 *
 * @thesis_note DE: Die Delete-Operation ermöglicht die Pflege des Wissens-
 * speichers. Im Benchmark SC-03 wird sie genutzt, um fehlerhafte Dokumente
 * zu entfernen und das Verhalten des Agenten nach der Bereinigung zu messen.
 */
export async function deleteDocument(collection: Collection, id: string): Promise<void> {
  await collection.delete({ ids: [id] });
}

// ---------------------------------------------------------------------------
// n8n Node class
// ---------------------------------------------------------------------------

/**
 * @description n8n custom node that provides native ChromaDB vector store
 * integration, eliminating the HTTP-workaround overhead of the standard n8n
 * HTTP Request node. Implements the retriever component of the RAG architecture
 * described by Lewis et al. (2020).
 *
 * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation for
 * Knowledge-Intensive NLP Tasks" - This node implements the document store and
 * retriever components of the RAG pipeline. Direct SDK usage reduces latency
 * compared to the HTTP-workaround baseline measured in the thesis benchmarks.
 *
 * @scientific_basis Waszkowski (2019) - "Low-code platform for automating
 * business processes" - Native ChromaDB integration via a dedicated n8n node
 * aligns with the low-code principle of encapsulating external service complexity
 * behind a visual node interface.
 *
 * @thesis_note DE: Der VectorStoreNode kapselt den gesamten RAG-Retrievalprozess
 * (Embedding + Speicherung / Suche) hinter einer einheitlichen n8n-Node-
 * Schnittstelle. Dies reduziert die Anzahl der benötigten HTTP-Request-Nodes
 * von 3 (Embedding + Store + Query) auf 1 und verbessert die Lesbarkeit des
 * Workflows erheblich.
 */
export class VectorStoreNode implements INodeType {
  description: INodeTypeDescription = {
    displayName: 'Vector Store (ChromaDB)',
    name: 'vectorStoreNode',
    icon: 'fa:database',
    group: ['transform'],
    version: 1,
    description:
      'Native ChromaDB integration for embedding, storing, and querying documents. ' +
      'Implements the RAG retriever component (Lewis et al., 2020).',
    defaults: {
      name: 'Vector Store',
      color: '#FF6B35',
    },
    inputs: ['main'],
    outputs: ['main'],
    credentials: [
      {
        name: 'chromaDbApi',
        required: false,
        displayOptions: { show: { operation: ['upsert', 'query', 'delete', 'listCollections'] } },
      },
    ],
    properties: [
      // ------------------------------------------------------------------
      // Operation selector
      // ------------------------------------------------------------------
      {
        displayName: 'Operation',
        name: 'operation',
        type: 'options',
        noDataExpression: true,
        options: [
          {
            name: 'Upsert',
            value: 'upsert',
            description: 'Embed text and store/update it in ChromaDB',
            action: 'Upsert a document into the vector store',
          },
          {
            name: 'Query',
            value: 'query',
            description: 'Semantic similarity search — return top-k results',
            action: 'Query the vector store for similar documents',
          },
          {
            name: 'Delete',
            value: 'delete',
            description: 'Remove a document from ChromaDB by ID',
            action: 'Delete a document from the vector store',
          },
          {
            name: 'List Collections',
            value: 'listCollections',
            description: 'List all collections in the ChromaDB instance',
            action: 'List all ChromaDB collections',
          },
        ],
        default: 'query',
      },

      // ------------------------------------------------------------------
      // Shared: ChromaDB connection
      // ------------------------------------------------------------------
      {
        displayName: 'ChromaDB URL',
        name: 'chromaUrl',
        type: 'string',
        default: 'http://localhost:8000',
        required: true,
        description: 'Base URL of the ChromaDB instance (e.g. http://localhost:8000)',
      },
      {
        displayName: 'Collection Name',
        name: 'collectionName',
        type: 'string',
        default: 'agent_knowledge',
        required: true,
        displayOptions: { show: { operation: ['upsert', 'query', 'delete'] } },
        description: 'Target ChromaDB collection name',
      },

      // ------------------------------------------------------------------
      // Upsert fields
      // ------------------------------------------------------------------
      {
        displayName: 'Document ID',
        name: 'documentId',
        type: 'string',
        default: '',
        required: true,
        displayOptions: { show: { operation: ['upsert', 'delete'] } },
        description: 'Unique identifier for the document (UUID recommended)',
      },
      {
        displayName: 'Text',
        name: 'text',
        type: 'string',
        typeOptions: { rows: 5 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['upsert'] } },
        description: 'Text content to embed and store',
      },
      {
        displayName: 'Metadata (JSON)',
        name: 'metadata',
        type: 'json',
        default: '{}',
        displayOptions: { show: { operation: ['upsert'] } },
        description: 'Optional JSON metadata associated with this document',
      },

      // ------------------------------------------------------------------
      // Query fields
      // ------------------------------------------------------------------
      {
        displayName: 'Query Text',
        name: 'queryText',
        type: 'string',
        typeOptions: { rows: 3 },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['query'] } },
        description: 'Natural-language query to search for similar documents',
      },
      {
        displayName: 'Top K Results',
        name: 'topK',
        type: 'number',
        default: 5,
        typeOptions: { minValue: 1, maxValue: 100 },
        displayOptions: { show: { operation: ['query'] } },
        description: 'Number of most similar documents to return (Lewis et al., 2020)',
      },

      // ------------------------------------------------------------------
      // Embedding model
      // ------------------------------------------------------------------
      {
        displayName: 'Embedding Model',
        name: 'embeddingModel',
        type: 'options',
        options: [
          { name: 'OpenAI text-embedding-3-small', value: 'text-embedding-3-small' },
          { name: 'OpenAI text-embedding-3-large', value: 'text-embedding-3-large' },
          { name: 'OpenAI text-embedding-ada-002', value: 'text-embedding-ada-002' },
        ],
        default: 'text-embedding-3-small',
        displayOptions: { show: { operation: ['upsert', 'query'] } },
        description: 'Embedding model to use for vectorising text',
      },
      {
        displayName: 'OpenAI API Key',
        name: 'openAiApiKey',
        type: 'string',
        typeOptions: { password: true },
        default: '',
        required: true,
        displayOptions: { show: { operation: ['upsert', 'query'] } },
        description: 'OpenAI API key for generating embeddings',
      },
    ],
  };

  /**
   * @description Main execution handler. Dispatches to the appropriate ChromaDB
   * operation based on the user-selected operation parameter.
   *
   * @scientific_basis Lewis et al. (2020) - "Retrieval-Augmented Generation" -
   * The execute method covers both the indexing phase (upsert) and the retrieval
   * phase (query) of the RAG pipeline within a single n8n node.
   *
   * @returns Standard n8n output item arrays.
   *
   * @thesis_note DE: Die execute-Methode kapselt den gesamten RAG-Lebenszyklus.
   * Durch die Zusammenfassung von Embedding-Generierung und ChromaDB-Zugriff in
   * einem einzigen Node wird der Benchmark-Vorteil gegenüber der HTTP-Baseline
   * (3 separate Nodes) messtechnisch erfasst.
   */
  async execute(this: IExecuteFunctions): Promise<INodeExecutionData[][]> {
    const items = this.getInputData();
    const returnData: INodeExecutionData[] = [];

    for (let i = 0; i < items.length; i++) {
      try {
        const operation = this.getNodeParameter('operation', i) as string;
        const chromaUrl = this.getNodeParameter('chromaUrl', i) as string;

        const client = new ChromaClient({ path: chromaUrl });

        switch (operation) {
          case 'upsert': {
            const collectionName = this.getNodeParameter('collectionName', i) as string;
            const documentId = this.getNodeParameter('documentId', i) as string;
            const text = this.getNodeParameter('text', i) as string;
            const metadataRaw = this.getNodeParameter('metadata', i) as string;
            const metadata = metadataRaw
              ? (JSON.parse(metadataRaw) as Record<string, unknown>)
              : {};
            const embeddingModel = this.getNodeParameter('embeddingModel', i) as string;
            const openAiApiKey = this.getNodeParameter('openAiApiKey', i) as string;

            const collection = await client.getOrCreateCollection({ name: collectionName });
            const id = await upsertDocument(
              collection,
              { id: documentId, text, metadata },
              openAiApiKey,
              embeddingModel,
            );

            returnData.push({
              json: { success: true, id, collectionName },
              pairedItem: { item: i },
            });
            break;
          }

          case 'query': {
            const collectionName = this.getNodeParameter('collectionName', i) as string;
            const queryText = this.getNodeParameter('queryText', i) as string;
            const topK = this.getNodeParameter('topK', i) as number;
            const embeddingModel = this.getNodeParameter('embeddingModel', i) as string;
            const openAiApiKey = this.getNodeParameter('openAiApiKey', i) as string;

            const collection = await client.getOrCreateCollection({ name: collectionName });
            const results = await queryDocuments(
              collection,
              queryText,
              topK,
              openAiApiKey,
              embeddingModel,
            );

            returnData.push({
              json: { results, count: results.length, collectionName },
              pairedItem: { item: i },
            });
            break;
          }

          case 'delete': {
            const collectionName = this.getNodeParameter('collectionName', i) as string;
            const documentId = this.getNodeParameter('documentId', i) as string;
            const collection = await client.getOrCreateCollection({ name: collectionName });
            await deleteDocument(collection, documentId);

            returnData.push({
              json: { success: true, deletedId: documentId, collectionName },
              pairedItem: { item: i },
            });
            break;
          }

          case 'listCollections': {
            const collections = await client.listCollections();
            returnData.push({
              json: { collections, count: collections.length },
              pairedItem: { item: i },
            });
            break;
          }

          default:
            throw new NodeOperationError(
              this.getNode(),
              `Unknown operation: ${operation}`,
              { itemIndex: i },
            );
        }
      } catch (error) {
        if (this.continueOnFail()) {
          returnData.push({
            json: { error: (error as Error).message },
            pairedItem: { item: i },
          });
          continue;
        }
        throw error;
      }
    }

    return [returnData];
  }
}
